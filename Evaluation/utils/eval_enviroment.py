#!/usr/bin/env python3
"""
Evaluator for the URBAN-ENVIRONMENTAL ASSESSMENT JSON schema.

What this evaluator does
------------------------
- Computes slot-level macro accuracy per section + a global macro accuracy.
- Uses teacher-gated scoring:
  * always scores ".visible"
  * scores ".value" / ".confidence" / ".target" / ".description" only if teacher says visible==True
- Scores list-valued fields with set-F1 (order-independent).
- Scores degradation_events with a bag-of-events metric (presence_f1 + count_acc + in_progress_acc).
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
    build_student_index,
    compute_teacher_run_consensus_weight,
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_confidence,
    is_str_list,
    read_json,
)
from utils.latex_table import write_latex_table
from utils.model_sort import format_model_name, sort_model_summary
from utils.run_paths import task_eval_dir, task_results_dir

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "environment"
GT_DIR = "/opt/dataset/ds_environment/test_dataset_json"#f"/opt/dataset/ds_{TASK_NAME}/test_dataset_json"
DEFAULT_RESULTS_DIR = task_results_dir(TASK_NAME)
DEFAULT_OUT_DIR = task_eval_dir(TASK_NAME)


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], Set[Any]] = {
    # waste_management
    ("waste_management", "illegal_dumping", "value"): {True, False},
    ("waste_management", "waste_location", "value"): {"sidewalk", "green_area", "roadside_layby", "unknown"},
    # urban_degradation
    ("urban_degradation", "graffiti_vandalism", "value"): {True, False},
    ("urban_degradation", "graffiti_vandalism", "target"): {"wall", "public_building", "unknown"},
    ("urban_degradation", "vehicle_vandalism", "value"): {True, False},
    ("urban_degradation", "infrastructure_damage", "broken_street_lights", "value"): {True, False},
    ("urban_degradation", "infrastructure_damage", "damaged_signage", "value"): {True, False},
    ("urban_degradation", "infrastructure_damage", "deteriorated_surfaces", "value"): {True, False},
    # environmental_pollution
    ("environmental_pollution", "visible_spills", "value"): {True, False},
    ("environmental_pollution", "air_smoke_emissions", "value"): {True, False},
    # degradation_events
    ("degradation_events", "*", "type"): {
        "littering_in_progress",
        "vandalism_act",
        "vehicle_damage_act",
        "structural_collapse",
        "fly_tipping",
    },
    ("degradation_events", "*", "impact_level"): {"minor", "moderate", "severe"},
    # aesthetic_and_safety_score
    ("aesthetic_and_safety_score", "degradation_index", "value"): {"negligible", "low", "moderate", "high"},
}

WASTE_TYPE_ENUM: Set[str] = {"household", "bulky", "hazardous", "unknown"}


# ----------------------------
# Slot paths (per-section)
# ----------------------------

def _indicator_paths(*path: str) -> List[Tuple[Any, ...]]:
    return [
        (*path, "visible"),
        (*path, "value"),
        (*path, "confidence"),
    ]


WASTE_SLOTS: List[Tuple[Any, ...]] = (
    _indicator_paths("waste_management", "illegal_dumping")
    + [
        ("waste_management", "waste_location", "value"),
        ("waste_management", "waste_location", "confidence"),
    ]
)

URBAN_DEGRADATION_SLOTS: List[Tuple[Any, ...]] = (
    _indicator_paths("urban_degradation", "graffiti_vandalism")
    + [("urban_degradation", "graffiti_vandalism", "target")]
    + _indicator_paths("urban_degradation", "vehicle_vandalism")
    + [("urban_degradation", "vehicle_vandalism", "description")]
    + _indicator_paths("urban_degradation", "infrastructure_damage", "broken_street_lights")
    + _indicator_paths("urban_degradation", "infrastructure_damage", "damaged_signage")
    + _indicator_paths("urban_degradation", "infrastructure_damage", "deteriorated_surfaces")
)

ENV_POLLUTION_SLOTS: List[Tuple[Any, ...]] = (
    _indicator_paths("environmental_pollution", "visible_spills")
    + _indicator_paths("environmental_pollution", "air_smoke_emissions")
)

AESTHETIC_SLOTS: List[Tuple[Any, ...]] = [
    ("aesthetic_and_safety_score", "degradation_index", "value"),
    ("aesthetic_and_safety_score", "degradation_index", "confidence"),
]

LIST_VALUE_PATHS = {
    "waste_type": ("waste_management", "waste_type"),
    "top_priority_issues": ("aesthetic_and_safety_score", "top_priority_issues"),
}


# ----------------------------
# Teacher gating for slot scoring
# ----------------------------

def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    Gating rules:
      - score all ...visible
      - score ...value / ...confidence / ...target / ...description only if the sibling visible==True in teacher
    """
    if not path:
        return True

    last = path[-1]
    if last == "visible":
        return True

    if last in {"value", "confidence", "target", "description"}:
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
# degradation_events scoring (bag-of-events)
# ----------------------------

@dataclass
class EventScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    in_progress_acc: float
    hits: List[float]


def _event_key(ev: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(ev, dict):
        return None
    et = ev.get("type")
    imp = ev.get("impact_level")
    if not (isinstance(et, str) and isinstance(imp, str)):
        return None
    return (et, imp)


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


def _event_in_progress_map(arr: Any) -> Dict[Tuple[str, str], bool]:
    out: Dict[Tuple[str, str], bool] = {}
    if not isinstance(arr, list):
        return out
    for ev in arr:
        if not isinstance(ev, dict):
            continue
        k = _event_key(ev)
        if k is None:
            continue
        inp = ev.get("in_progress")
        out[k] = bool(out.get(k, False) or (inp is True))
    return out


def score_degradation_events(student: Dict[str, Any], teacher: Dict[str, Any]) -> EventScores:
    s = student.get("degradation_events")
    t = teacher.get("degradation_events")

    s_counts = _event_counts(s)
    t_counts = _event_counts(t)

    s_keys = set(s_counts.keys())
    t_keys = set(t_counts.keys())

    if not t_keys:
        return EventScores(
            macro_acc=float("nan"),
            presence_f1=float("nan"),
            count_acc=float("nan"),
            in_progress_acc=float("nan"),
            hits=[],
        )

    presence_f1 = _set_f1(s_keys, t_keys)

    count_hits: List[float] = []
    for k in t_keys:
        count_hits.append(1.0 if s_counts.get(k, 0) == t_counts.get(k, 0) else 0.0)
    count_acc = float(np.mean(count_hits)) if count_hits else 1.0

    s_inp = _event_in_progress_map(s)
    t_inp = _event_in_progress_map(t)
    inprog_hits: List[float] = []
    for k in t_keys:
        inprog_hits.append(1.0 if bool(s_inp.get(k, False)) == bool(t_inp.get(k, False)) else 0.0)
    in_progress_acc = float(np.mean(inprog_hits)) if inprog_hits else 1.0

    hits = [presence_f1, count_acc, in_progress_acc]
    macro = float(np.mean(hits)) if hits else 0.0
    return EventScores(macro_acc=macro, presence_f1=presence_f1, count_acc=count_acc, in_progress_acc=in_progress_acc, hits=hits)


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    waste_management_macro_acc: float
    urban_degradation_macro_acc: float
    environmental_pollution_macro_acc: float
    degradation_events_macro_acc: float
    aesthetic_and_safety_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    waste = _score_leaf_slots(student, teacher, WASTE_SLOTS)
    urban = _score_leaf_slots(student, teacher, URBAN_DEGRADATION_SLOTS)
    env = _score_leaf_slots(student, teacher, ENV_POLLUTION_SLOTS)
    aesth = _score_leaf_slots(student, teacher, AESTHETIC_SLOTS)

    # list-valued fields
    waste_type_f1 = _score_list_f1(student, teacher, LIST_VALUE_PATHS["waste_type"])
    top_issues_f1 = _score_list_f1(student, teacher, LIST_VALUE_PATHS["top_priority_issues"])

    waste_hits = waste.hits + [waste_type_f1]
    aesth_hits = aesth.hits + [top_issues_f1]

    events = score_degradation_events(student, teacher)
    events_hits = events.hits

    all_hits = waste_hits + urban.hits + env.hits + events_hits + aesth_hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        waste_management_macro_acc=float(np.mean(waste_hits)) if waste_hits else 0.0,
        urban_degradation_macro_acc=urban.macro_acc,
        environmental_pollution_macro_acc=env.macro_acc,
        degradation_events_macro_acc=events.macro_acc,
        aesthetic_and_safety_macro_acc=float(np.mean(aesth_hits)) if aesth_hits else 0.0,
    )
    return sec, {
        "waste_management": waste_hits,
        "urban_degradation": urban.hits,
        "environmental_pollution": env.hits,
        "degradation_events": events_hits,
        "aesthetic_and_safety_score": aesth_hits,
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
    """
    Checks objects that must contain exactly (at least) "value" and "confidence" (no "visible" required),
    as per aesthetic_and_safety_score.degradation_index schema.
    """
    if not isinstance(obj, dict):
        errors.append(f"not_object:{tag}")
        return
    if "value" not in obj:
        errors.append(f"missing_key:{tag}.value")
    if "confidence" not in obj:
        errors.append(f"missing_key:{tag}.confidence")


def validate_struct(obj: Any) -> ValidationReport:
    # Point (3): ensure top-level is a JSON object/dict
    if obj is None or not isinstance(obj, dict):
        return ValidationReport(
            parse_ok=False,
            schema_ok=False,
            rule_ok=False,
            errors=["parse_failed_or_not_object"],
        )

    errors: List[str] = []

    required_top = [
        "waste_management",
        "urban_degradation",
        "environmental_pollution",
        "degradation_events",
        "aesthetic_and_safety_score",
    ]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    for sec in ["waste_management", "urban_degradation", "environmental_pollution", "aesthetic_and_safety_score"]:
        if sec in obj and not isinstance(obj.get(sec), dict):
            errors.append(f"not_object:{sec}")

    # waste_management
    _check_indicator_obj(get_path(obj, ("waste_management", "illegal_dumping")), "waste_management.illegal_dumping", errors)
    wl = get_path(obj, ("waste_management", "waste_location"))
    if not isinstance(wl, dict):
        errors.append("not_object:waste_management.waste_location")
    else:
        if "value" not in wl:
            errors.append("missing_key:waste_management.waste_location.value")
        if "confidence" not in wl:
            errors.append("missing_key:waste_management.waste_location.confidence")
    wt = get_path(obj, ("waste_management", "waste_type"))
    if wt is None:
        errors.append("missing_key:waste_management.waste_type")
    elif not is_str_list(wt):
        errors.append("bad_type:waste_management.waste_type")

    # urban_degradation
    _check_indicator_obj(get_path(obj, ("urban_degradation", "graffiti_vandalism")), "urban_degradation.graffiti_vandalism", errors)
    gv = get_path(obj, ("urban_degradation", "graffiti_vandalism"))
    if isinstance(gv, dict) and "target" not in gv:
        errors.append("missing_key:urban_degradation.graffiti_vandalism.target")

    _check_indicator_obj(get_path(obj, ("urban_degradation", "vehicle_vandalism")), "urban_degradation.vehicle_vandalism", errors)
    vv = get_path(obj, ("urban_degradation", "vehicle_vandalism"))
    if isinstance(vv, dict) and "description" not in vv:
        errors.append("missing_key:urban_degradation.vehicle_vandalism.description")

    inf = get_path(obj, ("urban_degradation", "infrastructure_damage"))
    if not isinstance(inf, dict):
        errors.append("not_object:urban_degradation.infrastructure_damage")
    _check_indicator_obj(get_path(obj, ("urban_degradation", "infrastructure_damage", "broken_street_lights")), "urban_degradation.infrastructure_damage.broken_street_lights", errors)
    _check_indicator_obj(get_path(obj, ("urban_degradation", "infrastructure_damage", "damaged_signage")), "urban_degradation.infrastructure_damage.damaged_signage", errors)
    _check_indicator_obj(get_path(obj, ("urban_degradation", "infrastructure_damage", "deteriorated_surfaces")), "urban_degradation.infrastructure_damage.deteriorated_surfaces", errors)

    # environmental_pollution
    _check_indicator_obj(get_path(obj, ("environmental_pollution", "visible_spills")), "environmental_pollution.visible_spills", errors)
    _check_indicator_obj(get_path(obj, ("environmental_pollution", "air_smoke_emissions")), "environmental_pollution.air_smoke_emissions", errors)

    # degradation_events
    de = obj.get("degradation_events")
    if de is None:
        errors.append("missing_key:degradation_events")
    elif not isinstance(de, list):
        # Point (2): make the tag match schema_ok predicate
        errors.append("degradation_events_not_list")

    # aesthetic_and_safety_score
    # Point (1): degradation_index is {value, confidence} (no visible)
    _check_value_conf_obj(
        get_path(obj, ("aesthetic_and_safety_score", "degradation_index")),
        "aesthetic_and_safety_score.degradation_index",
        errors,
    )
    tpi = get_path(obj, ("aesthetic_and_safety_score", "top_priority_issues"))
    if tpi is None:
        errors.append("missing_key:aesthetic_and_safety_score.top_priority_issues")
    elif not is_str_list(tpi):
        errors.append("bad_type:aesthetic_and_safety_score.top_priority_issues")

    # Confidence checks (rule_ok component)
    rule_ok = True
    confidence_paths = [
        ("waste_management", "illegal_dumping", "confidence"),
        ("waste_management", "waste_location", "confidence"),
        ("urban_degradation", "graffiti_vandalism", "confidence"),
        ("urban_degradation", "vehicle_vandalism", "confidence"),
        ("urban_degradation", "infrastructure_damage", "broken_street_lights", "confidence"),
        ("urban_degradation", "infrastructure_damage", "damaged_signage", "confidence"),
        ("urban_degradation", "infrastructure_damage", "deteriorated_surfaces", "confidence"),
        ("environmental_pollution", "visible_spills", "confidence"),
        ("environmental_pollution", "air_smoke_emissions", "confidence"),
        ("aesthetic_and_safety_score", "degradation_index", "confidence"),
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

    # degradation_events checks
    if isinstance(de, list):
        for i, ev in enumerate(de):
            if not isinstance(ev, dict):
                errors.append(f"degradation_events[{i}]_not_object")
                continue
            et = ev.get("type")
            if not isinstance(et, str):
                errors.append(f"degradation_events[{i}].type_missing_or_not_str")
            imp = ev.get("impact_level")
            if not isinstance(imp, str):
                errors.append(f"degradation_events[{i}].impact_level_missing_or_not_str")
            desc = ev.get("description")
            if not isinstance(desc, str):
                errors.append(f"degradation_events[{i}].description_missing_or_not_str")
            conf = ev.get("confidence")
            if conf is None:
                errors.append(f"missing_conf:degradation_events[{i}].confidence")
                rule_ok = False
            elif not is_confidence(conf):
                errors.append(f"bad_conf:degradation_events[{i}].confidence={conf}")
                rule_ok = False
            inp = ev.get("in_progress")
            if not isinstance(inp, bool):
                errors.append(f"degradation_events[{i}].in_progress_bad={inp}")

    # Enum checks (all defined ENUMS)
    for path, allowed in ENUMS.items():
        if path[0] == "degradation_events" and path[1] == "*":
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    # list enums
    if isinstance(wt, list):
        for i, v in enumerate(wt):
            if v not in WASTE_TYPE_ENUM:
                errors.append(f"bad_enum:waste_management.waste_type[{i}]={v}")

    # wildcard enum checks for degradation_events
    if isinstance(de, list):
        for i, ev in enumerate(de):
            if not isinstance(ev, dict):
                continue
            _check_enum(ev.get("type"), ENUMS[("degradation_events", "*", "type")], f"degradation_events[{i}].type", errors)
            _check_enum(ev.get("impact_level"), ENUMS[("degradation_events", "*", "impact_level")], f"degradation_events[{i}].impact_level", errors)

    # Point (2): fix schema_ok predicate to actually catch "degradation_events_not_list"
    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.startswith("missing_key:")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        or e.startswith("bad_type:")
        or e.endswith("_not_list")          # catches *_not_list style tags
        or e.endswith("not_list")           # catches "degradation_events_not_list"
        or e.endswith("_not_object")
        or "_missing_or_not_" in e
        for e in errors
    )

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# Teacher-run consensus weighting
# ----------------------------

def _sym_global_slot_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa, _ = score_global_and_sections(a, b)
    sb, _ = score_global_and_sections(b, a)
    return 0.5 * (sa.global_macro_acc + sb.global_macro_acc)


def _sym_degradation_events_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return 0.5 * (score_degradation_events(a, b).macro_acc + score_degradation_events(b, a).macro_acc)


def _consensus_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gpair = _sym_global_slot_acc(a, b)
    epair = _sym_degradation_events_acc(a, b)
    vals = [v for v in (gpair, epair) if not np.isnan(v)]
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
        "--skip-missing-students",
        action="store_true",
        help="Skip gold videos with zero matching student outputs.",
    )
    ap.add_argument(
        "--include-teacher-runs",
        action="store_true",
        help="Compute per-video consensus weights from teacher runs and report weighted means.",
    )
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return ap


def run(args: argparse.Namespace) -> None:
    if args.results == DEFAULT_RESULTS_DIR and not args.results.exists():
        legacy_results = BASE_DIR / "results_environment"
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
        if teacher is None or not isinstance(teacher, dict):
            log(f"Skipping teacher (parse failed or not object): {teacher_path}")
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
            waste_slot = None
            urban_slot = None
            env_slot = None
            events_slot = None
            aesth_slot = None
            events_presence_f1 = None
            events_count_acc = None
            events_in_progress_acc = None

            # Only score if validator says parse_ok
            if vrep.parse_ok:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                waste_slot = sec.waste_management_macro_acc
                urban_slot = sec.urban_degradation_macro_acc
                env_slot = sec.environmental_pollution_macro_acc
                events_slot = sec.degradation_events_macro_acc
                aesth_slot = sec.aesthetic_and_safety_macro_acc

                evs = score_degradation_events(student, teacher)
                events_presence_f1 = evs.presence_f1
                events_count_acc = evs.count_acc
                events_in_progress_acc = evs.in_progress_acc

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "waste_management_macro_acc": waste_slot,
                    "urban_degradation_macro_acc": urban_slot,
                    "environmental_pollution_macro_acc": env_slot,
                    "degradation_events_macro_acc": events_slot,
                    "aesthetic_and_safety_macro_acc": aesth_slot,
                    "events_presence_f1": events_presence_f1,
                    "events_count_acc": events_count_acc,
                    "events_in_progress_acc": events_in_progress_acc,
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
            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")

            agg_rows.append(
                {
                    "model": model_name,
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "waste_management_macro_acc_mean": mean_or_nan("waste_management_macro_acc"),
                    "urban_degradation_macro_acc_mean": mean_or_nan("urban_degradation_macro_acc"),
                    "environmental_pollution_macro_acc_mean": mean_or_nan("environmental_pollution_macro_acc"),
                    "degradation_events_macro_acc_mean": mean_or_nan("degradation_events_macro_acc"),
                    "aesthetic_and_safety_macro_acc_mean": mean_or_nan("aesthetic_and_safety_macro_acc"),
                    "events_presence_f1_mean": mean_or_nan("events_presence_f1"),
                    "events_count_acc_mean": mean_or_nan("events_count_acc"),
                    "events_in_progress_acc_mean": mean_or_nan("events_in_progress_acc"),
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
