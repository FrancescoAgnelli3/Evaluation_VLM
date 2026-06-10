#!/usr/bin/env python3
"""
Evaluator for the ROAD-SAFETY ASSESSMENT JSON schema.

Changes vs previous rewrite
---------------------------
1) "slot accuracy" is now a GLOBAL metric computed over *all scorable leaves* across:
   - road_composition
   - environmental_conditions
   - traffic_status
   - vehicle_events
   - pedestrian_events
   (Risk Assessment already has its own metrics and is kept separate.)

2) Adds one specific metric per section, in the same spirit as Risk Assessment metrics:
   - road_composition_slot_macro_acc
   - environmental_conditions_slot_macro_acc
   - traffic_status_macro_acc
   - vehicle_events_macro_acc
   - pedestrian_events_macro_acc
   - global_slot_macro_acc  (the global metric requested)

3) Keeps validation (parse_ok / schema_ok / rule_ok) best-effort.

Dependencies:
  pip install numpy pandas
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from utils.eval_common import (
    _set_f1,
    compute_teacher_run_consensus_weight,
    build_student_index,
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_bool_or_unknown,
    is_confidence,
    is_int_or_unknown,
    read_json,
    safe_float,
)
from utils.latex_table import write_latex_table
from utils.model_sort import format_model_name, sort_model_summary
from utils.run_paths import task_eval_dir, task_results_dir

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "road"
GT_DIR = f"/opt/dataset/ds_pulito/test_dataset_json"
DEFAULT_RESULTS_DIR = task_results_dir(TASK_NAME)
DEFAULT_OUT_DIR = task_eval_dir(TASK_NAME)


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], set] = {
    # road_composition
    ("road_composition", "lane_directions", "value"): {"one_way", "two_way", "unknown"},
    ("road_composition", "road_type", "value"): {"urban", "suburban", "highway", "pedestrian_zone", "unknown"},
    ("road_composition", "pedestrian_crossings", "type", "value"): {"zebra", "traffic_light_controlled", "other", "unknown"},
    ("road_composition", "traffic_lights", "visible_state", "value"): {"red", "yellow", "green", "unknown"},
    # environmental_conditions
    ("environmental_conditions", "weather", "value"): {"clear", "rain", "snow", "fog", "storm", "unknown"},
    ("environmental_conditions", "precipitation_intensity", "value"): {"none", "light", "moderate", "heavy", "unknown"},
    ("environmental_conditions", "ambient_visibility", "value"): {"good", "reduced", "poor", "unknown"},
    ("environmental_conditions", "road_surface_condition", "value"): {"dry", "wet", "icy", "snow_covered", "unknown"},
    ("environmental_conditions", "lighting_conditions", "value"): {"daylight", "dusk_dawn", "night", "artificial_lighting", "unknown"},
    ("environmental_conditions", "wind_conditions", "value"): {"calm", "windy", "strong_wind", "unknown"},
    # traffic_status
    ("traffic_status", "*", "status", "value"): {"free", "slow", "congested", "stopped"},
    ("traffic_status", "*", "vehicle_density", "value"): {"low", "medium", "high"},
    # vehicle_events
    ("vehicle_events", "*", "event_type"): {"speeding", "red_light_violation", "illegal_lane_change", "wrong_direction", "illegal_stop", "other"},
    ("vehicle_events", "*", "involved_vehicle_type"): {"car", "truck", "motorcycle", "bus", "unknown"},
    ("vehicle_events", "*", "risk_level"): {"low", "medium", "high"},
    # pedestrian_events
    ("pedestrian_events", "*", "behavior"): {"crossing_legally", "jaywalking", "waiting", "standing_in_road", "other"},
    ("pedestrian_events", "*", "location"): {"crosswalk", "sidewalk", "roadway", "median", "unknown"},
    ("pedestrian_events", "*", "interaction_with_vehicles"): {"none", "near_miss", "conflict"},
    ("pedestrian_events", "*", "risk_level"): {"low", "medium", "high"},
    # Risk Assessment overall
    ("Risk Assessment", "*", "overall_risk_assessment", "risk_level", "value"): {"low", "medium", "high"},
}

RISK_OBS_VEH_IDS = [
    "crossed_intersection",
    "motorcycle_without_helmet_observed",
    "lane_change_observed",
    "wrong_direction_observed",
    "blocking_stop_line_or_crosswalk_observed",
    "stopped_in_travel_lane_observed",
    "likely_high_speed_observed",
]
RISK_OBS_PAIRS = ["tailgating_observed"]


# ----------------------------
# Slot paths (per-section)
# ----------------------------

ROAD_COMPOSITION_SLOTS: List[Tuple[Any, ...]] = [
    ("road_composition", "number_of_lanes", "visible"),
    ("road_composition", "number_of_lanes", "value"),
    ("road_composition", "lane_directions", "visible"),
    ("road_composition", "lane_directions", "value"),
    ("road_composition", "road_type", "visible"),
    ("road_composition", "road_type", "value"),
    ("road_composition", "pedestrian_crossings", "visible"),
    ("road_composition", "pedestrian_crossings", "present", "value"),
    ("road_composition", "pedestrian_crossings", "type", "value"),
    ("road_composition", "traffic_lights", "visible"),
    ("road_composition", "traffic_lights", "present", "value"),
    ("road_composition", "traffic_lights", "visible_state", "value"),
    ("road_composition", "horizontal_signage", "visible"),
    ("road_composition", "horizontal_signage", "lane_markings", "value"),
    ("road_composition", "horizontal_signage", "stop_lines", "value"),
    ("road_composition", "horizontal_signage", "other_markings", "value"),
    ("road_composition", "vertical_signage", "visible"),
    ("road_composition", "vertical_signage", "speed_limit", "value"),
    ("road_composition", "vertical_signage", "warning_signs", "value"),
    ("road_composition", "vertical_signage", "prohibition_signs", "value"),
]

ENVIRONMENT_SLOTS: List[Tuple[Any, ...]] = [
    ("environmental_conditions", "weather", "visible"),
    ("environmental_conditions", "weather", "value"),
    ("environmental_conditions", "precipitation_intensity", "visible"),
    ("environmental_conditions", "precipitation_intensity", "value"),
    ("environmental_conditions", "ambient_visibility", "visible"),
    ("environmental_conditions", "ambient_visibility", "value"),
    ("environmental_conditions", "road_surface_condition", "visible"),
    ("environmental_conditions", "road_surface_condition", "value"),
    ("environmental_conditions", "lighting_conditions", "visible"),
    ("environmental_conditions", "lighting_conditions", "value"),
    ("environmental_conditions", "wind_conditions", "visible"),
    ("environmental_conditions", "wind_conditions", "value"),
]


def _pair_key(p: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(p, dict):
        return None
    a = p.get("leader_id")
    b = p.get("follower_id")
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    return (a, b)


# ----------------------------
# Teacher gating for slot scoring (road/env)
# ----------------------------

def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    For road_composition and environmental_conditions slots:
      - score all .visible
      - score .value only if teacher says the corresponding element is visible/present
    """
    if not path:
        return True

    if path[-1] == "visible":
        return True

    # road_composition.<field>.value gated by road_composition.<field>.visible
    if len(path) == 3 and path[0] == "road_composition" and path[-1] == "value":
        vis = get_path(teacher, (path[0], path[1], "visible"))
        return vis is True

    # environmental_conditions.<field>.value gated by environmental_conditions.<field>.visible
    if len(path) == 3 and path[0] == "environmental_conditions" and path[-1] == "value":
        vis = get_path(teacher, (path[0], path[1], "visible"))
        return vis is True

    # pedestrian_crossings.present/type gating
    if path[:2] == ("road_composition", "pedestrian_crossings") and path[-1] == "value":
        pc_vis = get_path(teacher, ("road_composition", "pedestrian_crossings", "visible"))
        if pc_vis is not True:
            return False
        if path[2] == "present":
            return True
        if path[2] == "type":
            present = get_path(teacher, ("road_composition", "pedestrian_crossings", "present", "value"))
            return present is True
        return True

    # traffic_lights.present/state gating
    if path[:2] == ("road_composition", "traffic_lights") and path[-1] == "value":
        tl_vis = get_path(teacher, ("road_composition", "traffic_lights", "visible"))
        if tl_vis is not True:
            return False
        if path[2] == "present":
            return True
        if path[2] == "visible_state":
            present = get_path(teacher, ("road_composition", "traffic_lights", "present", "value"))
            return present is True
        return True

    # signage subfields gated by signage.visible
    if path[:2] == ("road_composition", "horizontal_signage") and path[-1] == "value":
        vis = get_path(teacher, ("road_composition", "horizontal_signage", "visible"))
        return vis is True

    if path[:2] == ("road_composition", "vertical_signage") and path[-1] == "value":
        vis = get_path(teacher, ("road_composition", "vertical_signage", "visible"))
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
# traffic_status scoring
# ----------------------------

@dataclass
class TrafficStatusScores:
    macro_acc: float
    length_acc: float
    per_lane_macro_acc: float
    hits: List[float]


def _lane_map(ts: Any) -> Dict[int, Dict[str, Any]]:
    if not isinstance(ts, list):
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for it in ts:
        if not isinstance(it, dict):
            continue
        lid = it.get("lane_id")
        if isinstance(lid, int) and not isinstance(lid, bool):
            out[lid] = it
    return out


def score_traffic_status(student: Dict[str, Any], teacher: Dict[str, Any]) -> TrafficStatusScores:
    """
    Scores traffic_status only when teacher traffic_status is a list.
    If teacher traffic_status is null => score exact match on null vs list (length_acc),
    but per-lane fields are not applicable.
    """
    s_ts = student.get("traffic_status")
    t_ts = teacher.get("traffic_status")

    hits: List[float] = []

    # length/null accuracy (always scored)
    if t_ts is None:
        length_acc = 1.0 if s_ts is None else 0.0
        hits.append(length_acc)
        return TrafficStatusScores(macro_acc=float(np.mean(hits)), length_acc=length_acc, per_lane_macro_acc=0.0, hits=hits)

    if not isinstance(t_ts, list):
        # teacher malformed; keep metric conservative
        length_acc = 0.0
        hits.append(length_acc)
        return TrafficStatusScores(macro_acc=float(np.mean(hits)), length_acc=length_acc, per_lane_macro_acc=0.0, hits=hits)

    # teacher list => student should be list
    length_acc = 1.0 if isinstance(s_ts, list) and len(s_ts) == len(t_ts) else 0.0
    hits.append(length_acc)

    tmap = _lane_map(t_ts)
    smap = _lane_map(s_ts)

    # Score per teacher lane_id, exact match on status.value and vehicle_density.value
    lane_hits: List[float] = []
    for lane_id, titem in tmap.items():
        sitem = smap.get(lane_id, {})
        # lane_id presence (weakly informative, but makes missing lanes penalize)
        lane_hits.append(1.0 if lane_id in smap else 0.0)

        sv = get_path(sitem, ("status", "value"))
        tv = get_path(titem, ("status", "value"))
        lane_hits.append(1.0 if sv == tv else 0.0)

        svd = get_path(sitem, ("vehicle_density", "value"))
        tvd = get_path(titem, ("vehicle_density", "value"))
        lane_hits.append(1.0 if svd == tvd else 0.0)

    per_lane_macro = float(np.mean(lane_hits)) if lane_hits else 0.0
    hits.extend(lane_hits)

    return TrafficStatusScores(
        macro_acc=float(np.mean(hits)) if hits else 0.0,
        length_acc=length_acc,
        per_lane_macro_acc=per_lane_macro,
        hits=hits,
    )


# ----------------------------
# vehicle/pedestrian events scoring (bag-of-events)
# ----------------------------

@dataclass
class EventScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    hits: List[float]


def _event_key_vehicle(ev: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    if not isinstance(ev, dict):
        return None
    et = ev.get("event_type")
    vt = ev.get("involved_vehicle_type")
    rl = ev.get("risk_level")
    if not (isinstance(et, str) and isinstance(vt, str) and isinstance(rl, str)):
        return None
    return (et, vt, rl)


def _event_key_ped(ev: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    if not isinstance(ev, dict):
        return None
    bh = ev.get("behavior")
    loc = ev.get("location")
    rl = ev.get("risk_level")
    if not (isinstance(bh, str) and isinstance(loc, str) and isinstance(rl, str)):
        return None
    return (bh, loc, rl)


def _event_counts(arr: Any, key_fn) -> Dict[Any, int]:
    out: Dict[Any, int] = {}
    if not isinstance(arr, list):
        return out
    for ev in arr:
        if not isinstance(ev, dict):
            continue
        k = key_fn(ev)
        if k is None:
            continue
        c = ev.get("count")
        if not (isinstance(c, int) and not isinstance(c, bool)):
            # treat bad/missing count as 0 for scoring purposes
            c = 0
        out[k] = out.get(k, 0) + int(c)
    return out


def score_vehicle_events(student: Dict[str, Any], teacher: Dict[str, Any]) -> EventScores:
    s = student.get("vehicle_events")
    t = teacher.get("vehicle_events")

    s_counts = _event_counts(s, _event_key_vehicle)
    t_counts = _event_counts(t, _event_key_vehicle)

    s_keys = set(s_counts.keys())
    t_keys = set(t_counts.keys())

    if not t_keys:
        return EventScores(
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
    return EventScores(macro_acc=macro, presence_f1=presence_f1, count_acc=count_acc, hits=hits)


def score_pedestrian_events(student: Dict[str, Any], teacher: Dict[str, Any]) -> EventScores:
    s = student.get("pedestrian_events")
    t = teacher.get("pedestrian_events")

    s_counts = _event_counts(s, _event_key_ped)
    t_counts = _event_counts(t, _event_key_ped)

    s_keys = set(s_counts.keys())
    t_keys = set(t_counts.keys())

    if not t_keys:
        return EventScores(
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
    return EventScores(macro_acc=macro, presence_f1=presence_f1, count_acc=count_acc, hits=hits)


# ----------------------------
# Risk Assessment scoring (unchanged structure, still separate)
# ----------------------------

@dataclass
class RiskObsScores:
    macro_acc: float
    vehicle_ids_f1: float
    pairs_f1: float
    overall_risk_level_acc: float
    main_risk_factors_f1: float


def _first_ra(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ra = obj.get("Risk Assessment")
    if not isinstance(ra, list) or not ra:
        return None
    return ra[0] if isinstance(ra[0], dict) else None


def score_risk_assessment(student: Dict[str, Any], teacher: Dict[str, Any]) -> RiskObsScores:
    sra = _first_ra(student)
    tra = _first_ra(teacher)
    if sra is None or tra is None:
        return RiskObsScores(macro_acc=0.0, vehicle_ids_f1=0.0, pairs_f1=0.0, overall_risk_level_acc=0.0, main_risk_factors_f1=0.0)

    so = sra.get("observations")
    to = tra.get("observations")
    if not isinstance(so, dict) or not isinstance(to, dict):
        return RiskObsScores(macro_acc=0.0, vehicle_ids_f1=0.0, pairs_f1=0.0, overall_risk_level_acc=0.0, main_risk_factors_f1=0.0)

    per_field: List[float] = []

    def score_obs_key(k: str) -> None:
        # visible always
        per_field.append(1.0 if get_path(so, (k, "visible")) == get_path(to, (k, "visible")) else 0.0)

        # value gated by teacher visible==true
        if get_path(to, (k, "visible")) is True:
            per_field.append(1.0 if get_path(so, (k, "value")) == get_path(to, (k, "value")) else 0.0)

    for k in RISK_OBS_VEH_IDS:
        score_obs_key(k)
    for k in RISK_OBS_PAIRS:
        score_obs_key(k)

    macro_acc = float(np.mean(per_field)) if per_field else 0.0

    id_f1s: List[float] = []
    for k in RISK_OBS_VEH_IDS:
        if get_path(to, (k, "value")) is True:
            sid = get_path(so, (k, "vehicle_ids")) or []
            tid = get_path(to, (k, "vehicle_ids")) or []
            id_f1s.append(_set_f1(sid, tid))
    vehicle_ids_f1 = float(np.mean(id_f1s)) if id_f1s else 0.0

    pair_f1s: List[float] = []
    for k in RISK_OBS_PAIRS:
        if get_path(to, (k, "value")) is True:
            spairs = get_path(so, (k, "pairs")) or []
            tpairs = get_path(to, (k, "pairs")) or []
            spk = [pk for pk in (_pair_key(p) for p in spairs) if pk is not None]
            tpk = [pk for pk in (_pair_key(p) for p in tpairs) if pk is not None]
            pair_f1s.append(_set_f1(spk, tpk))
    pairs_f1 = float(np.mean(pair_f1s)) if pair_f1s else 0.0

    srl = get_path(sra, ("overall_risk_assessment", "risk_level", "value"))
    trl = get_path(tra, ("overall_risk_assessment", "risk_level", "value"))
    overall_risk_level_acc = 1.0 if (srl == trl) else 0.0

    smrf = get_path(sra, ("overall_risk_assessment", "main_risk_factors")) or []
    tmrf = get_path(tra, ("overall_risk_assessment", "main_risk_factors")) or []
    main_risk_factors_f1 = _set_f1(smrf, tmrf)

    return RiskObsScores(
        macro_acc=macro_acc,
        vehicle_ids_f1=vehicle_ids_f1,
        pairs_f1=pairs_f1,
        overall_risk_level_acc=overall_risk_level_acc,
        main_risk_factors_f1=main_risk_factors_f1,
    )


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    road_composition_macro_acc: float
    environmental_macro_acc: float
    traffic_status_macro_acc: float
    vehicle_events_macro_acc: float
    pedestrian_events_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    """
    Returns:
      - section/global macro scores
      - raw hit lists per section (used to compute global as a true macro over leaves)
    """
    road = _score_leaf_slots(student, teacher, ROAD_COMPOSITION_SLOTS)
    env = _score_leaf_slots(student, teacher, ENVIRONMENT_SLOTS)
    ts = score_traffic_status(student, teacher)
    ve = score_vehicle_events(student, teacher)
    pe = score_pedestrian_events(student, teacher)

    # Global = mean over all hits across all sections (true global leaf macro)
    all_hits = road.hits + env.hits + ts.hits + ve.hits + pe.hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        road_composition_macro_acc=road.macro_acc,
        environmental_macro_acc=env.macro_acc,
        traffic_status_macro_acc=ts.macro_acc,
        vehicle_events_macro_acc=ve.macro_acc,
        pedestrian_events_macro_acc=pe.macro_acc,
    )
    return sec, {
        "road": road.hits,
        "env": env.hits,
        "traffic_status": ts.hits,
        "vehicle_events": ve.hits,
        "pedestrian_events": pe.hits,
    }


# ----------------------------
# Validation (best-effort, kept similar)
# ----------------------------

@dataclass
class ValidationReport:
    parse_ok: bool
    schema_ok: bool
    rule_ok: bool
    errors: List[str]


def _check_enum(value: Any, allowed: set, tag: str, errors: List[str]) -> None:
    def _in_allowed(v: Any) -> bool:
        try:
            return v in allowed
        except TypeError:
            return False

    if value is None:
        errors.append(f"missing_enum:{tag}")
        return

    # Accept list/tuple/set by checking each element.
    if isinstance(value, (list, tuple, set)):
        if not value:
            errors.append(f"bad_enum:{tag}={value}")
            return
        for v in value:
            if not _in_allowed(v):
                errors.append(f"bad_enum:{tag}={v}")
        return

    if not _in_allowed(value):
        errors.append(f"bad_enum:{tag}={value}")


def validate_struct(obj: Optional[Dict[str, Any]]) -> ValidationReport:
    if obj is None:
        return ValidationReport(parse_ok=False, schema_ok=False, rule_ok=False, errors=["parse_failed_or_not_object"])

    errors: List[str] = []

    required_top = [
        "road_composition",
        "environmental_conditions",
        "traffic_status",
        "vehicle_events",
        "pedestrian_events",
        "Risk Assessment",
    ]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    rc = obj.get("road_composition")
    ec = obj.get("environmental_conditions")
    if not isinstance(rc, dict):
        errors.append("not_object:road_composition")
    if not isinstance(ec, dict):
        errors.append("not_object:environmental_conditions")

    ts = obj.get("traffic_status")
    if ts is not None and not isinstance(ts, list):
        errors.append("traffic_status_not_null_or_list")

    ve = obj.get("vehicle_events")
    pe = obj.get("pedestrian_events")
    if not isinstance(ve, list):
        errors.append("vehicle_events_not_list")
    if not isinstance(pe, list):
        errors.append("pedestrian_events_not_list")

    ra = obj.get("Risk Assessment")
    if not isinstance(ra, list):
        errors.append("risk_assessment_not_list")

    # Lane dependency rule
    rule_ok = True
    nol_vis = get_path(obj, ("road_composition", "number_of_lanes", "visible"))
    nol_val = get_path(obj, ("road_composition", "number_of_lanes", "value"))

    if nol_vis is True:
        if ts is None:
            errors.append("rule_lane:visible_true_but_traffic_status_null")
            rule_ok = False
        if not is_int_or_unknown(nol_val):
            errors.append(f"rule_lane:number_of_lanes.value_bad={nol_val}")
            rule_ok = False
        if isinstance(nol_val, int):
            if not isinstance(ts, list) or len(ts) != nol_val:
                got = 0 if ts is None else (len(ts) if isinstance(ts, list) else -1)
                errors.append(f"rule_lane:traffic_status_len_mismatch expected={nol_val} got={got}")
                rule_ok = False
    elif nol_vis is False:
        if ts is not None:
            errors.append("rule_lane:visible_false_but_traffic_status_not_null")
            rule_ok = False
    else:
        errors.append("missing_or_bad_visible:road_composition.number_of_lanes.visible")
        rule_ok = False

    # Confidence range checks (best-effort): scan a few known places + traffic_status and events
    # (This stays light; scoring does not use confidence.)
    def check_conf(path: Tuple[Any, ...], tag: str) -> None:
        v = get_path(obj, path)
        if v is None:
            errors.append(f"missing_conf:{tag}")
        elif not is_confidence(v):
            errors.append(f"bad_conf:{tag}={v}")

    # road/env main triplets
    for fld in ["number_of_lanes", "lane_directions", "road_type"]:
        check_conf(("road_composition", fld, "confidence"), f"road_composition.{fld}.confidence")
    for fld in ["weather", "precipitation_intensity", "ambient_visibility", "road_surface_condition", "lighting_conditions", "wind_conditions"]:
        check_conf(("environmental_conditions", fld, "confidence"), f"environmental_conditions.{fld}.confidence")

    # traffic_status per-lane confs
    if isinstance(ts, list):
        for i, it in enumerate(ts):
            if not isinstance(it, dict):
                errors.append(f"traffic_status[{i}]_not_object")
                continue
            check_conf(("traffic_status", i, "status", "confidence"), f"traffic_status[{i}].status.confidence")  # will fail due to int indexing in get_path
            # do manual for list items
            sc = get_path(it, ("status", "confidence"))
            if sc is None:
                errors.append(f"missing_conf:traffic_status[{i}].status.confidence")
            elif not is_confidence(sc):
                errors.append(f"bad_conf:traffic_status[{i}].status.confidence={sc}")

            dc = get_path(it, ("vehicle_density", "confidence"))
            if dc is None:
                errors.append(f"missing_conf:traffic_status[{i}].vehicle_density.confidence")
            elif not is_confidence(dc):
                errors.append(f"bad_conf:traffic_status[{i}].vehicle_density.confidence={dc}")

    # event confs
    def check_events_conf(arr: Any, kind: str) -> None:
        if not isinstance(arr, list):
            return
        for i, evv in enumerate(arr):
            if not isinstance(evv, dict):
                continue
            c = evv.get("confidence")
            if c is None:
                errors.append(f"missing_conf:{kind}[{i}].confidence")
            elif not is_confidence(c):
                errors.append(f"bad_conf:{kind}[{i}].confidence={c}")

    check_events_conf(ve, "vehicle_events")
    check_events_conf(pe, "pedestrian_events")

    # Enum checks (non-wildcard)
    for path, allowed in ENUMS.items():
        if "*" in path:
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    # Enum checks (wildcard)
    if isinstance(ts, list):
        for i, it in enumerate(ts):
            if not isinstance(it, dict):
                continue
            _check_enum(get_path(it, ("status", "value")), ENUMS[("traffic_status", "*", "status", "value")], f"traffic_status[{i}].status.value", errors)
            _check_enum(
                get_path(it, ("vehicle_density", "value")),
                ENUMS[("traffic_status", "*", "vehicle_density", "value")],
                f"traffic_status[{i}].vehicle_density.value",
                errors,
            )

    if isinstance(ve, list):
        for i, evv in enumerate(ve):
            if not isinstance(evv, dict):
                continue
            _check_enum(evv.get("event_type"), ENUMS[("vehicle_events", "*", "event_type")], f"vehicle_events[{i}].event_type", errors)
            _check_enum(evv.get("involved_vehicle_type"), ENUMS[("vehicle_events", "*", "involved_vehicle_type")], f"vehicle_events[{i}].involved_vehicle_type", errors)
            _check_enum(evv.get("risk_level"), ENUMS[("vehicle_events", "*", "risk_level")], f"vehicle_events[{i}].risk_level", errors)

    if isinstance(pe, list):
        for i, evv in enumerate(pe):
            if not isinstance(evv, dict):
                continue
            _check_enum(evv.get("behavior"), ENUMS[("pedestrian_events", "*", "behavior")], f"pedestrian_events[{i}].behavior", errors)
            _check_enum(evv.get("location"), ENUMS[("pedestrian_events", "*", "location")], f"pedestrian_events[{i}].location", errors)
            _check_enum(evv.get("interaction_with_vehicles"), ENUMS[("pedestrian_events", "*", "interaction_with_vehicles")], f"pedestrian_events[{i}].interaction_with_vehicles", errors)
            _check_enum(evv.get("risk_level"), ENUMS[("pedestrian_events", "*", "risk_level")], f"pedestrian_events[{i}].risk_level", errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.endswith("_not_list")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        for e in errors
    )

    if any(e.startswith("bad_conf:") or e.startswith("rule_lane:") for e in errors):
        rule_ok = False

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# Teacher-run consensus weighting
# ----------------------------

def _sym_global_slot_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa, _ = score_global_and_sections(a, b)
    sb, _ = score_global_and_sections(b, a)
    return 0.5 * (sa.global_macro_acc + sb.global_macro_acc)


def _sym_risk_obs_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return 0.5 * (score_risk_assessment(a, b).macro_acc + score_risk_assessment(b, a).macro_acc)


def _consensus_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gpair = _sym_global_slot_acc(a, b)
    rpair = _sym_risk_obs_acc(a, b)
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
        "--no-train",
        action="store_true",
        help="Evaluate only test-set items whose video_id begins with the held-out list.",
    )
    ap.add_argument(
        "--all-train",
        action="store_true",
        help="Evaluate all test-set items except those whose video_id begins with the held-out list.",
    )
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
        legacy_results = BASE_DIR / "results_road"
        if legacy_results.exists():
            args.results = legacy_results
    if args.no_train and args.all_train:
        raise SystemExit("Cannot use --no-train and --all-train together.")

    heldout_prefixes = {
        "329",
        "330",
        "331",
        "333",
        "334",
        "335",
        "336",
        "337",
        "338",
        "339",
        "340",
        "341",
        "342",
        "343",
        "345",
        "346",
        "347",
        "353",
        "354",
        "355",
    }

    if args.no_train:
        if args.out == DEFAULT_OUT_DIR:
            args.out = DEFAULT_OUT_DIR / "no_train"
    elif args.all_train:
        if args.out == DEFAULT_OUT_DIR:
            args.out = DEFAULT_OUT_DIR / "all_train"

    args.out.mkdir(parents=True, exist_ok=True)
    verbose = bool(args.verbose)
    df_agg_out = args.out / "model_summary.csv"

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    if df_agg_out.exists():
        log(f"Overwriting existing summary: {df_agg_out}")

    teacher_standards = discover_teacher_standards(args.results_gold)
    if args.no_train:
        teacher_standards = [
            (vid, p) for (vid, p) in teacher_standards if any(vid.startswith(hp) for hp in heldout_prefixes)
        ]
    elif args.all_train:
        teacher_standards = [
            (vid, p) for (vid, p) in teacher_standards if not any(vid.startswith(hp) for hp in heldout_prefixes)
        ]
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
            road_slot = None
            env_slot = None
            ts_slot = None
            veh_events_slot = None
            ped_events_slot = None

            risk_obs_macro = None
            risk_vehicle_ids_f1 = None
            risk_pairs_f1 = None
            overall_risk_level_acc = None
            main_risk_factors_f1 = None

            if vrep.parse_ok:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                road_slot = sec.road_composition_macro_acc
                env_slot = sec.environmental_macro_acc
                ts_slot = sec.traffic_status_macro_acc
                veh_events_slot = sec.vehicle_events_macro_acc
                ped_events_slot = sec.pedestrian_events_macro_acc

                ra = score_risk_assessment(student, teacher)
                risk_obs_macro = ra.macro_acc
                risk_vehicle_ids_f1 = ra.vehicle_ids_f1
                risk_pairs_f1 = ra.pairs_f1
                overall_risk_level_acc = ra.overall_risk_level_acc
                main_risk_factors_f1 = ra.main_risk_factors_f1

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "road_composition_slot_macro_acc": road_slot,
                    "environmental_conditions_slot_macro_acc": env_slot,
                    "traffic_status_macro_acc": ts_slot,
                    "vehicle_events_macro_acc": veh_events_slot,
                    "pedestrian_events_macro_acc": ped_events_slot,
                    "risk_obs_macro_acc": risk_obs_macro,
                    "risk_vehicle_ids_f1": risk_vehicle_ids_f1,
                    "risk_pairs_f1": risk_pairs_f1,
                    "overall_risk_level_acc": overall_risk_level_acc,
                    "main_risk_factors_f1": main_risk_factors_f1,
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
            risk_obs_mean = mean_or_nan("risk_obs_macro_acc")
            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")

            agg_rows.append(
                {
                    "model": model_name,
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "road_composition_slot_macro_acc_mean": mean_or_nan("road_composition_slot_macro_acc"),
                    "environmental_conditions_slot_macro_acc_mean": mean_or_nan("environmental_conditions_slot_macro_acc"),
                    "traffic_status_macro_acc_mean": mean_or_nan("traffic_status_macro_acc"),
                    "vehicle_events_macro_acc_mean": mean_or_nan("vehicle_events_macro_acc"),
                    "pedestrian_events_macro_acc_mean": mean_or_nan("pedestrian_events_macro_acc"),
                    "risk_obs_macro_acc_mean": risk_obs_mean,
                    "overall_risk_level_acc_mean": mean_or_nan("overall_risk_level_acc"),
                    "main_risk_factors_f1_mean": mean_or_nan("main_risk_factors_f1"),
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
