#!/usr/bin/env python3
"""
Integrate (aggregate) multiple Gemini perception runs into a single JSON per video.

This script expects files produced by create_gold_standard_gemini.py, named like:
  <video_stem>.teacher.perception_raw.run_1.json
  <video_stem>.teacher.perception_raw.run_2.json
  ...

For each <video_stem>, it:
- loads all run JSONs
- keeps only non-error JSON runs
- aggregates them with aggregate_perception_runs(...) while keeping the same schema/shape
- writes:
    <video_stem>.teacher.perception_integrated.json

Usage:
  python aggregate_gold_standard.py \
    --in-dir /path/to/results_gold \
    --min-valid 3 \
    --write-risks

Note: This file does not validate against a schema; it assumes the prompt format.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "results_gold"
RAW_RE = re.compile(r"^(?P<stem>.+)\.teacher\.perception_raw\.run_(?P<run>\d+)\.json$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *.teacher.perception_raw.run_*.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write integrated outputs (default: --in-dir)")
    ap.add_argument("--min-valid", type=int, default=1, help="Minimum number of valid runs required to write an integrated JSON")
    ap.add_argument("--write-risks", action="store_true", help="Also write deterministic risks + overall risk for integrated perception")
    return ap.parse_args()


# ----------------------------
# Aggregation utilities (perception-level)
# ----------------------------

def _median(xs: List[float]) -> float:
    xs2 = [float(x) for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    return float(statistics.median(xs2)) if xs2 else 0.0


def _vote_str(xs: List[str]) -> str:
    xs2 = [x for x in xs if isinstance(x, str) and x.strip() != ""]
    return Counter(xs2).most_common(1)[0][0] if xs2 else "unknown"


def _vote_bool(xs: List[bool]) -> bool:
    xs2 = [x for x in xs if isinstance(x, bool)]
    if not xs2:
        return False
    c = Counter(xs2)
    if c[True] == c[False]:
        return False
    return c[True] > c[False]


def _vote_int_or_unknown(xs: List[Union[int, str, None]]) -> Union[int, str]:
    vals: List[Union[int, str]] = []
    for x in xs:
        if isinstance(x, int):
            vals.append(x)
        elif isinstance(x, str) and x == "unknown":
            vals.append("unknown")
    if not vals:
        return "unknown"
    return Counter(vals).most_common(1)[0][0]


def _vote_bool_or_unknown(xs: List[Union[bool, str, None]]) -> Union[bool, str]:
    vals: List[Union[bool, str]] = []
    for x in xs:
        if isinstance(x, bool):
            vals.append(x)
        elif isinstance(x, str) and x == "unknown":
            vals.append("unknown")
    if not vals:
        return "unknown"
    return Counter(vals).most_common(1)[0][0]


def _get_in(obj: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def aggregate_perception_runs(valid_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate perception runs into a single perception JSON dict.
    Strategy:
    - vote for booleans / enums / unknown
    - median for confidences
    - for traffic_status: aggregate per lane_id
    - for vehicle_events/pedestrian_events: aggregate per event key with voted fields
    - for Risk Assessment: aggregate observations and overall risk assessment
    """
    runs = list(valid_runs)
    if not runs:
        return {}

    rc_runs = [r["road_composition"] for r in runs if isinstance(r.get("road_composition"), dict)]
    env_runs = [r["environmental_conditions"] for r in runs if isinstance(r.get("environmental_conditions"), dict)]

    def agg_visible_scalar(path: List[str]) -> Dict[str, Any]:
        vis = [_get_in(r, path + ["visible"]) for r in rc_runs]
        val = [_get_in(r, path + ["value"]) for r in rc_runs]
        conf = [_get_in(r, path + ["confidence"]) for r in rc_runs]
        return {
            "visible": _vote_bool([bool(x) for x in vis if isinstance(x, bool)]),
            "value": _vote_int_or_unknown(val),
            "confidence": _median([float(x) for x in conf if isinstance(x, (int, float))]),
        }

    def agg_visible_enum(path: List[str]) -> Dict[str, Any]:
        vis = [_get_in(r, path + ["visible"]) for r in rc_runs]
        val = [_get_in(r, path + ["value"]) for r in rc_runs]
        conf = [_get_in(r, path + ["confidence"]) for r in rc_runs]
        return {
            "visible": _vote_bool([bool(x) for x in vis if isinstance(x, bool)]),
            "value": _vote_str([str(x) for x in val if isinstance(x, str)]),
            "confidence": _median([float(x) for x in conf if isinstance(x, (int, float))]),
        }

    def agg_present_bool(path: List[str]) -> Dict[str, Any]:
        val = [_get_in(r, path + ["value"]) for r in rc_runs]
        conf = [_get_in(r, path + ["confidence"]) for r in rc_runs]
        return {
            "value": _vote_bool([bool(x) for x in val if isinstance(x, bool)]),
            "confidence": _median([float(x) for x in conf if isinstance(x, (int, float))]),
        }

    def agg_typed_value(path: List[str]) -> Dict[str, Any]:
        val = [_get_in(r, path + ["value"]) for r in rc_runs]
        conf = [_get_in(r, path + ["confidence"]) for r in rc_runs]
        return {
            "value": _vote_str([str(x) for x in val if isinstance(x, str)]),
            "confidence": _median([float(x) for x in conf if isinstance(x, (int, float))]),
        }

    road_composition = {
        "number_of_lanes": agg_visible_scalar(["number_of_lanes"]),
        "lane_directions": agg_visible_enum(["lane_directions"]),
        "road_type": agg_visible_enum(["road_type"]),
        "pedestrian_crossings": {
            "visible": _vote_bool([_get_in(r, ["pedestrian_crossings", "visible"]) for r in rc_runs if isinstance(_get_in(r, ["pedestrian_crossings", "visible"]), bool)]),
            "present": agg_present_bool(["pedestrian_crossings", "present"]),
            "type": agg_typed_value(["pedestrian_crossings", "type"]),
        },
        "traffic_lights": {
            "visible": _vote_bool([_get_in(r, ["traffic_lights", "visible"]) for r in rc_runs if isinstance(_get_in(r, ["traffic_lights", "visible"]), bool)]),
            "present": agg_present_bool(["traffic_lights", "present"]),
            "visible_state": agg_typed_value(["traffic_lights", "visible_state"]),
        },
        "horizontal_signage": {
            "visible": _vote_bool([_get_in(r, ["horizontal_signage", "visible"]) for r in rc_runs if isinstance(_get_in(r, ["horizontal_signage", "visible"]), bool)]),
            "lane_markings": agg_present_bool(["horizontal_signage", "lane_markings"]),
            "stop_lines": agg_present_bool(["horizontal_signage", "stop_lines"]),
            "other_markings": agg_present_bool(["horizontal_signage", "other_markings"]),
        },
        "vertical_signage": {
            "visible": _vote_bool([_get_in(r, ["vertical_signage", "visible"]) for r in rc_runs if isinstance(_get_in(r, ["vertical_signage", "visible"]), bool)]),
            "speed_limit": {
                "value": _vote_int_or_unknown([_get_in(r, ["vertical_signage", "speed_limit", "value"]) for r in rc_runs]),
                "confidence": _median(
                    [
                        float(x)
                        for x in [_get_in(r, ["vertical_signage", "speed_limit", "confidence"]) for r in rc_runs]
                        if isinstance(x, (int, float))
                    ]
                ),
            },
            "warning_signs": agg_present_bool(["vertical_signage", "warning_signs"]),
            "prohibition_signs": agg_present_bool(["vertical_signage", "prohibition_signs"]),
        },
    }

    def agg_env_enum(key: str) -> Dict[str, Any]:
        vis = [_get_in(r, [key, "visible"]) for r in env_runs]
        val = [_get_in(r, [key, "value"]) for r in env_runs]
        conf = [_get_in(r, [key, "confidence"]) for r in env_runs]
        return {
            "visible": _vote_bool([bool(x) for x in vis if isinstance(x, bool)]),
            "value": _vote_str([str(x) for x in val if isinstance(x, str)]),
            "confidence": _median([float(x) for x in conf if isinstance(x, (int, float))]),
        }

    environmental_conditions = {
        "weather": agg_env_enum("weather"),
        "precipitation_intensity": agg_env_enum("precipitation_intensity"),
        "ambient_visibility": agg_env_enum("ambient_visibility"),
        "road_surface_condition": agg_env_enum("road_surface_condition"),
        "lighting_conditions": agg_env_enum("lighting_conditions"),
        "wind_conditions": agg_env_enum("wind_conditions"),
    }

    # traffic_status
    num_lanes = road_composition["number_of_lanes"]
    lanes_visible = bool(num_lanes.get("visible", False))
    ts_runs = [r.get("traffic_status") for r in runs]

    if not lanes_visible:
        traffic_status_out = None
    else:
        by_lane: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in ts_runs:
            if not isinstance(t, list):
                continue
            for lane in t:
                if not isinstance(lane, dict):
                    continue
                try:
                    lane_id = int(lane.get("lane_id"))
                except Exception:
                    continue
                by_lane[lane_id].append(lane)

        traffic_status_out = []
        for lane_id, items in sorted(by_lane.items(), key=lambda x: x[0]):
            status_vals = [it.get("status", {}).get("value") for it in items]
            status_confs = [it.get("status", {}).get("confidence") for it in items]
            dens_vals = [it.get("vehicle_density", {}).get("value") for it in items]
            dens_confs = [it.get("vehicle_density", {}).get("confidence") for it in items]
            traffic_status_out.append(
                {
                    "lane_id": lane_id,
                    "status": {
                        "value": _vote_str([str(x) for x in status_vals if isinstance(x, str)]),
                        "confidence": _median([float(x) for x in status_confs if isinstance(x, (int, float))]),
                    },
                    "vehicle_density": {
                        "value": _vote_str([str(x) for x in dens_vals if isinstance(x, str)]),
                        "confidence": _median([float(x) for x in dens_confs if isinstance(x, (int, float))]),
                    },
                }
            )

    # vehicle_events: aggregate per (event_type, involved_vehicle_type)
    event_runs = [r.get("vehicle_events") for r in runs]
    event_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for evs in event_runs:
        if not isinstance(evs, list):
            continue
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            event_type = str(ev.get("event_type", "")).strip()
            veh_type = str(ev.get("involved_vehicle_type", "")).strip()
            if not event_type:
                continue
            event_groups[(event_type, veh_type)].append(ev)

    vehicle_events_out: List[Dict[str, Any]] = []
    for (event_type, veh_type), items in sorted(event_groups.items(), key=lambda x: (x[0][0], x[0][1])):
        desc_vals = [it.get("description") for it in items]
        risk_vals = [it.get("risk_level") for it in items]
        conf_vals = [it.get("confidence") for it in items]
        count_vals = [it.get("count") for it in items]
        count_med = _median([float(x) for x in count_vals if isinstance(x, (int, float))])
        vehicle_events_out.append(
            {
                "event_type": event_type,
                "description": _vote_str([str(x) for x in desc_vals if isinstance(x, str)]),
                "involved_vehicle_type": veh_type,
                "risk_level": _vote_str([str(x) for x in risk_vals if isinstance(x, str)]),
                "confidence": _median([float(x) for x in conf_vals if isinstance(x, (int, float))]),
                "count": int(round(count_med)) if count_vals else 0,
            }
        )

    # pedestrian_events: aggregate per (behavior, location, interaction_with_vehicles)
    ped_runs = [r.get("pedestrian_events") for r in runs]
    ped_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for evs in ped_runs:
        if not isinstance(evs, list):
            continue
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            behavior = str(ev.get("behavior", "")).strip()
            location = str(ev.get("location", "")).strip()
            interaction = str(ev.get("interaction_with_vehicles", "")).strip()
            if not behavior:
                continue
            ped_groups[(behavior, location, interaction)].append(ev)

    pedestrian_events_out: List[Dict[str, Any]] = []
    for (behavior, location, interaction), items in sorted(ped_groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        risk_vals = [it.get("risk_level") for it in items]
        conf_vals = [it.get("confidence") for it in items]
        count_vals = [it.get("count") for it in items]
        count_med = _median([float(x) for x in count_vals if isinstance(x, (int, float))])
        pedestrian_events_out.append(
            {
                "behavior": behavior,
                "location": location,
                "interaction_with_vehicles": interaction,
                "risk_level": _vote_str([str(x) for x in risk_vals if isinstance(x, str)]),
                "confidence": _median([float(x) for x in conf_vals if isinstance(x, (int, float))]),
                "count": int(round(count_med)) if count_vals else 0,
            }
        )

    # Risk Assessment: aggregate observations + overall risk assessment
    ra_runs: List[Dict[str, Any]] = []
    for r in runs:
        ra = r.get("Risk Assessment")
        if isinstance(ra, list) and ra:
            if isinstance(ra[0], dict):
                ra_runs.append(ra[0])

    obs_runs = [r.get("observations", {}) for r in ra_runs]
    obs_keys: List[str] = []
    for o in obs_runs:
        if isinstance(o, dict):
            for k in o.keys():
                if k not in obs_keys:
                    obs_keys.append(k)

    def _merge_vehicle_ids(values: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for v in values:
            vs = v.get("vehicle_ids")
            if isinstance(vs, list):
                ids.extend([str(x) for x in vs])
        seen: set[str] = set()
        ordered: List[str] = []
        for vid in ids:
            if vid in seen:
                continue
            seen.add(vid)
            ordered.append(vid)
        return ordered

    def _merge_pairs(values: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        for v in values:
            ps = v.get("pairs")
            if isinstance(ps, list):
                for p in ps:
                    if isinstance(p, dict) and "leader_id" in p and "follower_id" in p:
                        pairs.append({"leader_id": str(p["leader_id"]), "follower_id": str(p["follower_id"])})
        seen = set()
        uniq_pairs = []
        for p in pairs:
            k2 = (p["leader_id"], p["follower_id"])
            if k2 in seen:
                continue
            seen.add(k2)
            uniq_pairs.append(p)
        return uniq_pairs

    observations_out: Dict[str, Any] = {}
    for key in obs_keys:
        entries = [o.get(key) for o in obs_runs if isinstance(o, dict) and isinstance(o.get(key), dict)]
        vis = [e.get("visible") for e in entries]
        val = [e.get("value") for e in entries]
        conf = [e.get("confidence") for e in entries]
        visible = _vote_bool([bool(x) for x in vis if isinstance(x, bool)])
        value = _vote_bool_or_unknown(val)
        confidence = _median([float(x) for x in conf if isinstance(x, (int, float))])

        entry_out: Dict[str, Any] = {"visible": visible, "value": value, "confidence": confidence}
        if entries and any("pairs" in e for e in entries):
            entry_out["pairs"] = _merge_pairs(entries) if value is True else []
        else:
            entry_out["vehicle_ids"] = _merge_vehicle_ids(entries) if value is True else []
        observations_out[key] = entry_out

    risk_runs = [r.get("overall_risk_assessment", {}) for r in ra_runs]
    risk_vals = [_get_in(r, ["risk_level", "value"]) for r in risk_runs]
    risk_confs = [_get_in(r, ["risk_level", "confidence"]) for r in risk_runs]
    factors: List[str] = []
    for r in risk_runs:
        fs = r.get("main_risk_factors")
        if isinstance(fs, list):
            factors.extend([str(x) for x in fs])
    seen_factors: set[str] = set()
    uniq_factors: List[str] = []
    for f in factors:
        if f in seen_factors:
            continue
        seen_factors.add(f)
        uniq_factors.append(f)

    risk_assessment_out = []
    if obs_keys or risk_runs:
        risk_assessment_out.append(
            {
                "observations": observations_out,
                "overall_risk_assessment": {
                    "risk_level": {
                        "value": _vote_str([str(x) for x in risk_vals if isinstance(x, str)]),
                        "confidence": _median([float(x) for x in risk_confs if isinstance(x, (int, float))]),
                    },
                    "main_risk_factors": uniq_factors,
                },
            }
        )

    aggregated = {
        "road_composition": road_composition,
        "environmental_conditions": environmental_conditions,
        "traffic_status": traffic_status_out,
        "vehicle_events": vehicle_events_out,
        "pedestrian_events": pedestrian_events_out,
        "Risk Assessment": risk_assessment_out,
    }

    return aggregated




def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_error_payload(d: Dict[str, Any]) -> bool:
    # Heuristic: generator writes {"error": "..."} or {"stage": "...", "error": "..."} on failure.
    return ("error" in d) and ("road_composition" not in d)


def main() -> None:
    args = parse_args()
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect run files by video stem
    groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for p in sorted(in_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        m = RAW_RE.match(p.name)
        if not m:
            continue
        stem = m.group("stem")
        run_i = int(m.group("run"))
        groups[stem].append((run_i, p))

    if not groups:
        raise SystemExit(f"No matching raw run files found in: {in_dir}")

    for stem, run_files in sorted(groups.items(), key=lambda x: x[0]):
        run_files = sorted(run_files, key=lambda x: x[0])

        valid_runs: List[Dict[str, Any]] = []
        invalid_info: List[Dict[str, Any]] = []

        for run_i, p in run_files:
            try:
                d = _load_json(p)
            except Exception as e:
                invalid_info.append({"run": run_i, "file": p.name, "error": f"read_failed: {type(e).__name__}: {e}"})
                continue

            if not isinstance(d, dict) or _is_error_payload(d):
                invalid_info.append({"run": run_i, "file": p.name, "error": "error_payload_or_not_object"})
                continue

            valid_runs.append(d)

        integrated_path = out_dir / f"{stem}.teacher.perception_integrated.json"

        if len(valid_runs) < args.min_valid:
            integrated_path.write_text(
                json.dumps(
                    {
                        "video_stem": stem,
                        "status": "insufficient_valid_runs",
                        "min_valid": args.min_valid,
                        "valid_runs": len(valid_runs),
                        "total_runs_found": len(run_files),
                        "invalid": invalid_info,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            continue

        # Aggregate
        aggregated = aggregate_perception_runs(valid_runs)

        # Save integrated perception
        integrated_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")



if __name__ == "__main__":
    main()
