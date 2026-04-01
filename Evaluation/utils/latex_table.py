from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _escape_latex(text: str) -> str:
    # Normalize escaped underscores from model names like "\\_" -> "_"
    text = text.replace("\\_", "_")
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("^", "\\textasciicircum{}")
        .replace("~", "\\textasciitilde{}")
    )


def _format_value(val: float, max_val: Optional[float], second_val: Optional[float], float_fmt: int) -> str:
    if pd.isna(val):
        return ""
    rendered = f"{float(val):.{float_fmt}f}"
    if max_val is not None and float(val) == max_val:
        return f"\\textbf{{{rendered}}}"
    if second_val is not None and float(val) == second_val:
        return f"\\underline{{{rendered}}}"
    return rendered


_SHORT_NAMES = {
    "parse_rate": "Parse",
    "schema_rate": "Schema",
    "rule_rate": "Rule",
    "global_slot_macro_acc_mean": "Global",
    "effective_global_slot_macro_acc_mean": "EffGlobal",
    "road_composition_slot_macro_acc_mean": "RoadComp",
    "environmental_conditions_slot_macro_acc_mean": "EnvCond",
    "traffic_status_macro_acc_mean": "Traffic",
    "vehicle_events_macro_acc_mean": "VehEvt",
    "pedestrian_events_macro_acc_mean": "PedEvt",
    "risk_obs_macro_acc_mean": "RiskObs",
    "overall_risk_level_acc_mean": "RiskLvl",
    "main_risk_factors_f1_mean": "RiskFacF1",
    "waste_management_macro_acc_mean": "Waste",
    "urban_degradation_macro_acc_mean": "UrbanDeg",
    "environmental_pollution_macro_acc_mean": "EnvPoll",
    "degradation_events_macro_acc_mean": "DegEvt",
    "aesthetic_and_safety_macro_acc_mean": "AesthSafe",
    "events_presence_f1_mean": "EvtPresF1",
    "events_count_acc_mean": "EvtCount",
    "events_in_progress_acc_mean": "EvtInProg",
    "site_characterization_slot_macro_acc_mean": "SiteChar",
    "personnel_safety_slot_macro_acc_mean": "PersSafe",
    "hazardous_conditions_slot_macro_acc_mean": "HazCond",
    "safety_violations_macro_acc_mean": "SafetyViol",
    "overall_site_risk_macro_acc_mean": "SiteRisk",
    "risk_rating_acc_mean": "RiskRate",
    "primary_risk_factors_f1_mean": "PrimRiskF1",
    "scene_context_slot_macro_acc_mean": "SceneCtx",
    "crowd_analysis_slot_macro_acc_mean": "Crowd",
    "violence_and_aggression_slot_macro_acc_mean": "ViolAgg",
    "specific_events_macro_acc_mean": "SpecEvt",
    "events_persons_count_acc_mean": "EvtPersons",
    "events_severity_acc_mean": "EvtSev",
    "risk_assessment_macro_acc_mean": "RiskAssess",
    "effective_risk_assessment_macro_acc_mean": "EffRisk",
    "overall_threat_level_acc_mean": "ThreatLvl",
    "main_threat_factors_f1_mean": "ThreatFacF1",
    "immediate_intervention_required_acc_mean": "IntervReq",
    "cleanliness_and_waste_macro_acc_mean": "CleanWaste",
    "infrastructure_condition_macro_acc_mean": "InfraCond",
    "environmental_indicators_macro_acc_mean": "EnvInd",
    "observed_activities_macro_acc_mean": "ObsAct",
    "events_macro_acc_mean": "Events",
    "uncertainty_summary_slot_macro_acc_mean": "Uncert",
}


def _abbrev_col(col: str) -> str:
    return _SHORT_NAMES.get(col, col)


def write_latex_table(
    df: pd.DataFrame,
    out_path: Path,
    float_fmt: int = 4,
    model_col: str = "model",
    abbreviate_headers: bool = True,
) -> None:
    if df.empty:
        out_path.write_text("", encoding="utf-8")
        return

    cols = [c for c in df.columns]
    if model_col not in cols:
        raise ValueError(f"Expected column '{model_col}' in dataframe")

    metric_cols = [c for c in cols if c != model_col]

    # Compute max and second-best per column (descending).
    best_map = {}
    second_map = {}
    for col in metric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            best_map[col] = None
            second_map[col] = None
            continue
        unique_vals = sorted({float(v) for v in series}, reverse=True)
        best_map[col] = unique_vals[0]
        second_map[col] = unique_vals[1] if len(unique_vals) > 1 else None

    align = "l" + ("r" * len(metric_cols))
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline"]

    if abbreviate_headers:
        header_cells = [_escape_latex(model_col)] + [_escape_latex(_abbrev_col(c)) for c in metric_cols]
    else:
        header_cells = [_escape_latex(model_col)] + [_escape_latex(c) for c in metric_cols]
    line_end = r" \\\\"
    lines.append(" & ".join(header_cells) + line_end)
    lines.append("\\hline")

    for _idx, row in df.iterrows():
        model_val = _escape_latex(str(row[model_col]))
        cells = [model_val]
        for col in metric_cols:
            val = pd.to_numeric(row[col], errors="coerce")
            cell = _format_value(val, best_map[col], second_map[col], float_fmt)
            cells.append(cell)
        lines.append(" & ".join(cells) + line_end)

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
