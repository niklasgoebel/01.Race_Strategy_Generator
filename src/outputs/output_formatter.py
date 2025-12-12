# src/outputs/output_formatter.py

from __future__ import annotations
import pandas as pd
from typing import Dict, Any


# -----------------------------
# Course overview
# -----------------------------

def make_course_overview_table(course_summary: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "distance_km": course_summary["total_distance_km"],
                "total_gain_m": course_summary["total_gain_m"],
                "total_loss_m": course_summary["total_loss_m"],
                "num_segments": course_summary["num_segments"],
            }
        ]
    )


# -----------------------------
# Segments
# -----------------------------

def make_segments_table(seg_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "type",
        "start_km",
        "end_km",
        "distance_km",
        "gain_m",
        "loss_m",
        "avg_gradient",
    ]

    out = seg_df[cols].copy()
    out = out.rename(columns={"avg_gradient": "avg_gradient_pct"})
    return out


# -----------------------------
# Key climbs
# -----------------------------

def make_key_climbs_table(key_climbs_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "type",
        "start_km",
        "end_km",
        "distance_km",
        "gain_m",
        "avg_gradient",
    ]

    out = key_climbs_df[cols].copy()
    out = out.rename(columns={"avg_gradient": "avg_gradient_pct"})
    return out


# -----------------------------
# Strategy tables
# -----------------------------

def make_strategy_tables(strategy_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Converts structured LLM output into DataFrames
    """
    tables = {}

    if "pacing_chunks" in strategy_data:
        tables["pacing_chunks"] = pd.DataFrame(strategy_data["pacing_chunks"])

    if "critical_sections" in strategy_data:
        tables["critical_sections"] = pd.DataFrame(strategy_data["critical_sections"])

    if "fueling_plan" in strategy_data:
        tables["fueling_plan"] = pd.DataFrame(
            [strategy_data["fueling_plan"]]
        )

    if "mental_cues" in strategy_data:
        tables["mental_cues"] = pd.DataFrame(strategy_data["mental_cues"])

    return tables