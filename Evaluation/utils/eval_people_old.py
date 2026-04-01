#!/usr/bin/env python3
"""
Evaluator for the URBAN PUBLIC-SPACE SAFETY-RELEVANT JSON schema.

Scoring philosophy (mirrors the previous road-safety evaluator style)
--------------------------------------------------------------------
1) "Slot accuracy" is a GLOBAL metric computed over *all scorable leaves* across:
   - scene_context
   - crowd_and_flow
   - observable_interactions
   - objects_of_interest
   - uncertainty_summary
   plus a dedicated section metric for events.

2) Per-section macro metrics are also reported:
   - scene_context_slot_macro_acc
   - crowd_and_flow_slot_macro_acc
   - observable_interactions_slot_macro_acc
   - objects_of_interest_slot_macro_acc
   - uncertainty_summary_slot_macro_acc
   - events_macro_acc
   - global_slot_macro_acc  (true macro over all scored leaves across sections)

3) Epistemic gating:
   - score all ".visible"
   - score ".value" only if teacher says the corresponding element is visible==true
   - for list/dict "value" fields we score set-based or structured metrics (see below)

Events scoring (bag-of-events)
------------------------------
- presence_f1 over event_type (set-based, counts ignored)
- count_acc over event_type counts (exact)
- persons_count_acc over event_type -> approx_involved_persons_count (exact per event instance aggregated by type)
- span_acc over event_type -> (start,end) exact per event instance aggregated by type

Text fields:
- event.description is NOT scored (too brittle)
- unknown_objects[].description IS scored as set-based matching (still brittle, but often short/controlled)
- arrays of strings (identified_harm_instruments, key_ambiguities, data_quality_issues) scored as set-F1.

Teacher-run consensus (optional)
--------------------------------
If --include-teacher-runs is enabled, computes per-video consensus weights from teacher runs only,
and reports weighted means for:
  - weight_global_slot_macro_acc_mean
  - weight_events_macro_acc_mean

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
    is_bool_or_unknown,
    is_confidence,
    is_int_or_unknown,
    read_json,
    safe_float,
    weighted_mean,
)
from utils.latex_table import write_latex_table
from utils.run_paths import task_eval_dir, task_results_dir

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "people"
GT_DIR = f"/opt/dataset/ds_{TASK_NAME}/test_dataset_json"
DEFAULT_RESULTS_DIR = task_results_dir(TASK_NAME)
DEFAULT_OUT_DIR = task_eval_dir(TASK_NAME)


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], set] = {
    ("scene_context", "location_type", "value"): {"street", "square", "park", "indoor_public", "transit_hub", "unknown"},
    ("scene_context", "lighting", "value"): {"daylight", "low_light", "night", "unknown"},
    ("scene_context", "weather_visibility", "value"): {"clear", "rain", "snow", "fog", "unknown"},
    ("scene_context", "camera_view", "value"): {"fixed", "panning", "zooming", "unknown"},
    ("crowd_and_flow", "movement_pattern", "value"): {"mostly_static", "walking_flow", "running", "dispersing", "converging", "unknown"},
    ("crowd_and_flow", "density_level", "value"): {"low", "medium", "high", "unknown"},
    ("observable_interactions", "contact_type_if_any", "value"): {"none", "accidental_brush", "pushing", "grabbing", "striking", "restraining", "unknown"},
    ("events", "*", "event_type"): {
        "unspecified_interaction",
        "congestion",
        "pushing_episode",
        "fall",
        "property_damage_visible",
        "medical_distress_visible",
        "unknown",
    },
}


# ----------------------------
# Slot paths (per-section)
# ----------------------------

SCENE_CONTEXT_SLOTS: List[Tuple[Any, ...]] = [
    ("scene_context", "location_type", "visible"),
    ("scene_context", "location_type", "value"),
    ("scene_context", "lighting", "visible"),
    ("scene_context", "lighting", "value"),
    ("scene_context", "weather_visibility", "visible"),
    ("scene_context", "weather_visibility", "value"),
    ("scene_context", "camera_view", "visible"),
    ("scene_context", "camera_view", "value"),
]

CROWD_FLOW_SLOTS: List[Tuple[Any, ...]] = [
    ("crowd_and_flow", "gathering_present", "visible"),
    ("crowd_and_flow", "gathering_present", "value"),
    ("crowd_and_flow", "estimated_count", "visible"),
    ("crowd_and_flow", "estimated_count", "value"),
    ("crowd_and_flow", "movement_pattern", "visible"),
    ("crowd_and_flow", "movement_pattern", "value"),
    ("crowd_and_flow", "density_level", "visible"),
    ("crowd_and_flow", "density_level", "value"),
]

INTERACTIONS_SLOTS: List[Tuple[Any, ...]] = [
    ("observable_interactions", "physical_contact_present", "visible"),
    ("observable_interactions", "physical_contact_present", "value"),
    ("observable_interactions", "contact_type_if_any", "visible"),
    ("observable_interactions", "contact_type_if_any", "value"),
    ("observable_interactions", "verbal_exchange_apparent", "visible"),
    ("observable_interactions", "verbal_exchange_apparent", "value"),
    ("observable_interactions", "high_amplitude_gestures_visible", "visible"),
    ("observable_interactions", "high_amplitude_gestures_visible", "value"),
]

# objects_of_interest are partly list/scalar hybrid; we score scalars as leaf slots,
# and handle list/dict-valued "value" via special scorers.
OBJECTS_SCALAR_SLOTS: List[Tuple[Any, ...]] = [
    ("objects_of_interest", "harm_instrument_clearly_identifiable", "visible"),
    ("objects_of_interest", "harm_instrument_clearly_identifiable", "value"),
    ("objects_of_interest", "identified_harm_instruments", "visible"),
    ("objects_of_interest", "unknown_objects", "visible"),
]

UNCERTAINTY_SCALAR_SLOTS: List[Tuple[Any, ...]] = [
    ("uncertainty_summary", "key_ambiguities", "visible"),
    ("uncertainty_summary", "data_quality_issues", "visible"),
]

# Dedicated list-valued paths (scored with set-F1 / structured scoring)
LIST_VALUE_PATHS = {
    "identified_harm_instruments": ("objects_of_interest", "identified_harm_instruments", "value"),
    "unknown_objects_descriptions": ("objects_of_interest", "unknown_objects", "value"),  # list of dicts
    "key_ambiguities": ("uncertainty_summary", "key_ambiguities", "value"),
    "data_quality_issues": ("uncertainty_summary", "data_quality_issues", "value"),
}


# ----------------------------
# Teacher gating for leaf slot scoring
# ----------------------------

def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    General rule:
      - score all ...visible
      - score ...value only if corresponding ...visible is True in teacher
    """
    if not path:
        return True

    if path[-1] == "visible":
        return True

    # Any <section>.<field>.value gated by <section>.<field>.visible when pattern matches
    if len(path) == 3 and path[-1] == "value":
        sec, field, _ = path
        vis = get_path(teacher, (sec, field, "visible"))
        return vis is True

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


def score_list_of_strings_value(student: Dict[str, Any], teacher: Dict[str, Any], base_path: Tuple[Any, ...]) -> ListValueScores:
    """
    Scores list-valued 'value' as set-F1, gated by corresponding visible==True in teacher.
    """
    # derive visible path from (..., "value") -> (..., "visible")
    if len(base_path) < 3 or base_path[-1] != "value":
        return ListValueScores(macro_acc=0.0, f1=0.0, hits=[])

    vis_path = base_path[:-1] + ("visible",)
    if get_path(teacher, vis_path) is not True:
        return ListValueScores(macro_acc=0.0, f1=0.0, hits=[])

    s_val = get_path(student, base_path)
    t_val = get_path(teacher, base_path)

    s_list = s_val if isinstance(s_val, list) else []
    t_list = t_val if isinstance(t_val, list) else []
    f1 = _set_f1(s_list, t_list)
    hits = [f1]
    return ListValueScores(macro_acc=float(np.mean(hits)), f1=f1, hits=hits)


def _unknown_object_descriptions(arr: Any) -> List[str]:
    if not isinstance(arr, list):
        return []
    out: List[str] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        d = it.get("description")
        if isinstance(d, str):
            out.append(d)
    return out


def score_unknown_objects_descriptions(student: Dict[str, Any], teacher: Dict[str, Any]) -> ListValueScores:
    """
    Scores unknown_objects.value as set-F1 over descriptions, gated by unknown_objects.visible in teacher.
    """
    vis_t = get_path(teacher, ("objects_of_interest", "unknown_objects", "visible"))
    if vis_t is not True:
        return ListValueScores(macro_acc=0.0, f1=0.0, hits=[])

    s_val = get_path(student, ("objects_of_interest", "unknown_objects", "value"))
    t_val = get_path(teacher, ("objects_of_interest", "unknown_objects", "value"))

    s_desc = _unknown_object_descriptions(s_val)
    t_desc = _unknown_object_descriptions(t_val)
    f1 = _set_f1(s_desc, t_desc)
    hits = [f1]
    return ListValueScores(macro_acc=float(np.mean(hits)), f1=f1, hits=hits)


# ----------------------------
# Events scoring (bag-of-events)
# ----------------------------

@dataclass
class EventScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    persons_count_acc: float
    span_acc: float
    hits: List[float]


def _event_key(ev: Dict[str, Any]) -> Optional[str]:
    if not isinstance(ev, dict):
        return None
    et = ev.get("event_type")
    return et if isinstance(et, str) else None


def _event_span(ev: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    ts = ev.get("time_span_in_frames")
    if not isinstance(ts, dict):
        return None
    s = ts.get("start_frame_index")
    e = ts.get("end_frame_index")
    if isinstance(s, int) and not isinstance(s, bool) and isinstance(e, int) and not isinstance(e, bool):
        return (s, e)
    return None


def _event_persons_count(ev: Dict[str, Any]) -> Any:
    return ev.get("approx_involved_persons_count")


def _events_by_type(arr: Any) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(arr, list):
        return out
    for ev in arr:
        if not isinstance(ev, dict):
            continue
        k = _event_key(ev)
        if k is None:
            continue
        out.setdefault(k, []).append(ev)
    return out


def score_events(student: Dict[str, Any], teacher: Dict[str, Any]) -> EventScores:
    s = student.get("events")
    t = teacher.get("events")

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
            span_acc=float("nan"),
            hits=[],
        )

    presence_f1 = _set_f1(s_types, t_types)

    # count accuracy per type (exact)
    count_hits: List[float] = []
    for et in t_types:
        count_hits.append(1.0 if len(s_map.get(et, [])) == len(t_map.get(et, [])) else 0.0)
    count_acc = float(np.mean(count_hits)) if count_hits else 1.0

    # persons_count_acc: compare multiset of counts per type (exact as strings)
    pc_hits: List[float] = []
    for et in t_types:
        s_counts = [str(_event_persons_count(ev)) for ev in s_map.get(et, [])]
        t_counts = [str(_event_persons_count(ev)) for ev in t_map.get(et, [])]
        pc_hits.append(1.0 if sorted(s_counts) == sorted(t_counts) else 0.0)
    persons_count_acc = float(np.mean(pc_hits)) if pc_hits else 1.0

    # span_acc: compare multiset of spans per type (exact)
    span_hits: List[float] = []
    for et in t_types:
        s_spans = [str(_event_span(ev)) for ev in s_map.get(et, [])]
        t_spans = [str(_event_span(ev)) for ev in t_map.get(et, [])]
        span_hits.append(1.0 if sorted(s_spans) == sorted(t_spans) else 0.0)
    span_acc = float(np.mean(span_hits)) if span_hits else 1.0

    hits = [presence_f1, count_acc, persons_count_acc, span_acc]
    macro = float(np.mean(hits)) if hits else 0.0
    return EventScores(
        macro_acc=macro,
        presence_f1=presence_f1,
        count_acc=count_acc,
        persons_count_acc=persons_count_acc,
        span_acc=span_acc,
        hits=hits,
    )


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    scene_context_macro_acc: float
    crowd_and_flow_macro_acc: float
    observable_interactions_macro_acc: float
    objects_of_interest_macro_acc: float
    uncertainty_summary_macro_acc: float
    events_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    """
    Returns:
      - section/global macro scores
      - raw hit lists per section (used to compute global as a true macro over leaves)
    """
    sc = _score_leaf_slots(student, teacher, SCENE_CONTEXT_SLOTS)
    cf = _score_leaf_slots(student, teacher, CROWD_FLOW_SLOTS)
    oi = _score_leaf_slots(student, teacher, INTERACTIONS_SLOTS)
    obj_scalar = _score_leaf_slots(student, teacher, OBJECTS_SCALAR_SLOTS)
    unc_scalar = _score_leaf_slots(student, teacher, UNCERTAINTY_SCALAR_SLOTS)

    # List-valued extras (each contributes one hit, gated)
    harm = score_list_of_strings_value(student, teacher, LIST_VALUE_PATHS["identified_harm_instruments"])
    unk = score_unknown_objects_descriptions(student, teacher)
    amb = score_list_of_strings_value(student, teacher, LIST_VALUE_PATHS["key_ambiguities"])
    dqi = score_list_of_strings_value(student, teacher, LIST_VALUE_PATHS["data_quality_issues"])

    ev = score_events(student, teacher)

    # Compose per-section hits
    scene_hits = sc.hits
    crowd_hits = cf.hits
    interaction_hits = oi.hits

    objects_hits = obj_scalar.hits + harm.hits + unk.hits
    uncertainty_hits = unc_scalar.hits + amb.hits + dqi.hits
    events_hits = ev.hits

    all_hits = scene_hits + crowd_hits + interaction_hits + objects_hits + uncertainty_hits + events_hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        scene_context_macro_acc=sc.macro_acc,
        crowd_and_flow_macro_acc=cf.macro_acc,
        observable_interactions_macro_acc=oi.macro_acc,
        objects_of_interest_macro_acc=float(np.mean(objects_hits)) if objects_hits else 0.0,
        uncertainty_summary_macro_acc=float(np.mean(uncertainty_hits)) if uncertainty_hits else 0.0,
        events_macro_acc=ev.macro_acc,
    )
    return sec, {
        "scene_context": scene_hits,
        "crowd_and_flow": crowd_hits,
        "observable_interactions": interaction_hits,
        "objects_of_interest": objects_hits,
        "uncertainty_summary": uncertainty_hits,
        "events": events_hits,
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
        "crowd_and_flow",
        "observable_interactions",
        "objects_of_interest",
        "events",
        "uncertainty_summary",
    ]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    # type checks
    if not isinstance(obj.get("scene_context"), dict):
        errors.append("not_object:scene_context")
    if not isinstance(obj.get("crowd_and_flow"), dict):
        errors.append("not_object:crowd_and_flow")
    if not isinstance(obj.get("observable_interactions"), dict):
        errors.append("not_object:observable_interactions")
    if not isinstance(obj.get("objects_of_interest"), dict):
        errors.append("not_object:objects_of_interest")
    if not isinstance(obj.get("uncertainty_summary"), dict):
        errors.append("not_object:uncertainty_summary")

    ev = obj.get("events")
    if not isinstance(ev, list):
        errors.append("events_not_list")

    # confidence checks: scan dicts that look like {visible, value, confidence}
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

    for fld in ["location_type", "lighting", "weather_visibility", "camera_view"]:
        check_triplet("scene_context", fld)
    for fld in ["gathering_present", "estimated_count", "movement_pattern", "density_level"]:
        check_triplet("crowd_and_flow", fld)
    for fld in ["physical_contact_present", "contact_type_if_any", "verbal_exchange_apparent", "high_amplitude_gestures_visible"]:
        check_triplet("observable_interactions", fld)
    for fld in ["harm_instrument_clearly_identifiable", "identified_harm_instruments", "unknown_objects"]:
        check_triplet("objects_of_interest", fld)
    for fld in ["key_ambiguities", "data_quality_issues"]:
        check_triplet("uncertainty_summary", fld)

    # basic value-type rules for a few fields
    est_count = get_path(obj, ("crowd_and_flow", "estimated_count", "value"))
    if not is_int_or_unknown(est_count):
        errors.append(f"bad_value:crowd_and_flow.estimated_count.value={est_count}")
        rule_ok = False

    # objects_of_interest list fields
    ih = get_path(obj, ("objects_of_interest", "identified_harm_instruments", "value"))
    if ih is not None and not isinstance(ih, list):
        errors.append("bad_type:objects_of_interest.identified_harm_instruments.value_not_list")
        rule_ok = False

    uo = get_path(obj, ("objects_of_interest", "unknown_objects", "value"))
    if uo is not None and not isinstance(uo, list):
        errors.append("bad_type:objects_of_interest.unknown_objects.value_not_list")
        rule_ok = False
    if isinstance(uo, list):
        for i, it in enumerate(uo):
            if not isinstance(it, dict):
                errors.append(f"unknown_objects[{i}]_not_object")
                rule_ok = False
                continue
            d = it.get("description")
            if d is None or not isinstance(d, str):
                errors.append(f"unknown_objects[{i}].description_missing_or_not_str")
                rule_ok = False
            c = it.get("confidence")
            if c is None:
                errors.append(f"missing_conf:unknown_objects[{i}].confidence")
                rule_ok = False
            elif not is_confidence(c):
                errors.append(f"bad_conf:unknown_objects[{i}].confidence={c}")
                rule_ok = False

    # uncertainty list fields
    ka = get_path(obj, ("uncertainty_summary", "key_ambiguities", "value"))
    if ka is not None and not isinstance(ka, list):
        errors.append("bad_type:uncertainty_summary.key_ambiguities.value_not_list")
        rule_ok = False
    dqi = get_path(obj, ("uncertainty_summary", "data_quality_issues", "value"))
    if dqi is not None and not isinstance(dqi, list):
        errors.append("bad_type:uncertainty_summary.data_quality_issues.value_not_list")
        rule_ok = False

    # events checks
    if isinstance(ev, list):
        for i, e in enumerate(ev):
            if not isinstance(e, dict):
                errors.append(f"events[{i}]_not_object")
                rule_ok = False
                continue
            et = e.get("event_type")
            if not isinstance(et, str):
                errors.append(f"events[{i}].event_type_missing_or_not_str")
                rule_ok = False
            pc = e.get("approx_involved_persons_count")
            if not is_int_or_unknown(pc):
                errors.append(f"events[{i}].approx_involved_persons_count_bad={pc}")
                rule_ok = False
            ts = e.get("time_span_in_frames")
            if not isinstance(ts, dict):
                errors.append(f"events[{i}].time_span_in_frames_not_object")
                rule_ok = False
            else:
                s = ts.get("start_frame_index")
                en = ts.get("end_frame_index")
                if not (isinstance(s, int) and not isinstance(s, bool)):
                    errors.append(f"events[{i}].time_span_in_frames.start_frame_index_bad={s}")
                    rule_ok = False
                if not (isinstance(en, int) and not isinstance(en, bool)):
                    errors.append(f"events[{i}].time_span_in_frames.end_frame_index_bad={en}")
                    rule_ok = False
            c = e.get("confidence")
            if c is None:
                errors.append(f"missing_conf:events[{i}].confidence")
                rule_ok = False
            elif not is_confidence(c):
                errors.append(f"bad_conf:events[{i}].confidence={c}")
                rule_ok = False
            # description existence (not scored, but schema says it exists)
            desc = e.get("description")
            if desc is None or not isinstance(desc, str):
                errors.append(f"events[{i}].description_missing_or_not_str")
                rule_ok = False

    # Enum checks (non-wildcard)
    for path, allowed in ENUMS.items():
        if "*" in path:
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    # Enum checks (wildcard) for events
    if isinstance(ev, list):
        for i, e in enumerate(ev):
            if not isinstance(e, dict):
                continue
            _check_enum(e.get("event_type"), ENUMS[("events", "*", "event_type")], f"events[{i}].event_type", errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.endswith("_not_list")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
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
    return 0.5 * (score_events(a, b).macro_acc + score_events(b, a).macro_acc)


def _consensus_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gpair = _sym_global_slot_acc(a, b)
    epair = _sym_events_acc(a, b)
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
            scene_slot = None
            crowd_slot = None
            interact_slot = None
            objects_slot = None
            uncertainty_slot = None
            events_macro = None

            events_presence_f1 = None
            events_count_acc = None
            events_persons_count_acc = None
            events_span_acc = None

            if vrep.parse_ok and "error" not in student:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                scene_slot = sec.scene_context_macro_acc
                crowd_slot = sec.crowd_and_flow_macro_acc
                interact_slot = sec.observable_interactions_macro_acc
                objects_slot = sec.objects_of_interest_macro_acc
                uncertainty_slot = sec.uncertainty_summary_macro_acc
                events_macro = sec.events_macro_acc

                evs = score_events(student, teacher)
                events_presence_f1 = evs.presence_f1
                events_count_acc = evs.count_acc
                events_persons_count_acc = evs.persons_count_acc
                events_span_acc = evs.span_acc

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": ("error" in student) if isinstance(student, dict) else False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "scene_context_slot_macro_acc": scene_slot,
                    "crowd_and_flow_slot_macro_acc": crowd_slot,
                    "observable_interactions_slot_macro_acc": interact_slot,
                    "objects_of_interest_slot_macro_acc": objects_slot,
                    "uncertainty_summary_slot_macro_acc": uncertainty_slot,
                    "events_macro_acc": events_macro,
                    "events_presence_f1": events_presence_f1,
                    "events_count_acc": events_count_acc,
                    "events_persons_count_acc": events_persons_count_acc,
                    "events_span_acc": events_span_acc,
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
            events_mean = mean_or_nan("events_macro_acc")
            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")

            weighted_global = float("nan")
            weighted_events = float("nan")
            if args.include_teacher_runs:
                weighted_global = weighted_mean(g2["global_slot_macro_acc"], g2["teacher_consensus_weight"])
                weighted_events = weighted_mean(g2["events_macro_acc"], g2["teacher_consensus_weight"])

            agg_rows.append(
                {
                    "model": model_name,
                    "n_videos": int(len(g2)),
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "scene_context_slot_macro_acc_mean": mean_or_nan("scene_context_slot_macro_acc"),
                    "crowd_and_flow_slot_macro_acc_mean": mean_or_nan("crowd_and_flow_slot_macro_acc"),
                    "observable_interactions_slot_macro_acc_mean": mean_or_nan("observable_interactions_slot_macro_acc"),
                    "objects_of_interest_slot_macro_acc_mean": mean_or_nan("objects_of_interest_slot_macro_acc"),
                    "uncertainty_summary_slot_macro_acc_mean": mean_or_nan("uncertainty_summary_slot_macro_acc"),
                    "events_macro_acc_mean": events_mean,
                    "events_presence_f1_mean": mean_or_nan("events_presence_f1"),
                    "events_count_acc_mean": mean_or_nan("events_count_acc"),
                    "events_persons_count_acc_mean": mean_or_nan("events_persons_count_acc"),
                    "events_span_acc_mean": mean_or_nan("events_span_acc"),
                    "weight_global_slot_macro_acc_mean": weighted_global,
                    "weight_events_macro_acc_mean": weighted_events,
                }
            )

    df_agg = pd.DataFrame(agg_rows)
    if not df_agg.empty:
        df_agg = df_agg.sort_values(
            by=["rule_rate", "global_slot_macro_acc_mean", "events_macro_acc_mean"],
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
