# src/outputs/chart_enhancements.py
"""
Enhanced elevation chart functionality with strategy overlays and markers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def effort_to_color(effort_rpe: str) -> str:
    """
    Convert RPE effort level to color for chart overlays.
    
    Args:
        effort_rpe: RPE string like "3-4", "5-6", etc.
    
    Returns:
        Color code (hex or CSS color name)
    """
    # Extract first number from RPE range
    try:
        if "-" in effort_rpe:
            rpe_val = int(effort_rpe.split("-")[0])
        else:
            rpe_val = int(effort_rpe.split()[0])
    except (ValueError, IndexError):
        return "rgba(128, 128, 128, 0.2)"  # Default gray
    
    # Color mapping based on RPE zones
    if rpe_val <= 3:
        return "rgba(76, 175, 80, 0.15)"  # Green - easy
    elif rpe_val <= 5:
        return "rgba(33, 150, 243, 0.15)"  # Blue - moderate
    elif rpe_val <= 7:
        return "rgba(255, 152, 0, 0.15)"  # Orange - hard
    else:
        return "rgba(244, 67, 54, 0.15)"  # Red - very hard


def create_strategy_overlay_data(
    segments_df: pd.DataFrame,
    strategy_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create overlay rectangles for effort zones on elevation chart.
    
    Args:
        segments_df: Course segments DataFrame
        strategy_data: Strategy data with pacing_chunks
    
    Returns:
        List of overlay dictionaries for Plotly
    """
    if not strategy_data or "pacing_chunks" not in strategy_data:
        return []
    
    overlays = []
    
    for chunk in strategy_data["pacing_chunks"]:
        start_km = float(chunk.get("start_km", 0))
        end_km = float(chunk.get("end_km", 0))
        effort_rpe = str(chunk.get("effort_rpe", ""))
        
        color = effort_to_color(effort_rpe)
        
        overlays.append({
            "x0": start_km,
            "x1": end_km,
            "color": color,
            "label": f"RPE {effort_rpe}",
        })
    
    return overlays


def generate_marker_positions(
    strategy_data: Dict[str, Any],
    df_gpx: pd.DataFrame
) -> Tuple[List[float], List[float], List[str], List[str]]:
    """
    Generate marker positions for fueling points and mental cues on elevation chart.
    
    Args:
        strategy_data: Strategy data with fueling_plan and mental_cues
        df_gpx: GPX DataFrame with elevation profile
    
    Returns:
        Tuple of (x_coords_km, y_coords_elev, labels, types)
    """
    if "cum_distance" not in df_gpx.columns:
        return [], [], [], []
    
    x_km = df_gpx["cum_distance"].to_numpy(dtype=float) / 1000.0
    
    # Use smoothed elevation if available
    if "elev_smooth" in df_gpx.columns:
        y = df_gpx["elev_smooth"].to_numpy(dtype=float)
    elif "elev_raw" in df_gpx.columns:
        y = df_gpx["elev_raw"].to_numpy(dtype=float)
    else:
        return [], [], [], []
    
    marker_x = []
    marker_y = []
    marker_labels = []
    marker_types = []
    
    # Fueling markers
    if strategy_data and "fueling_plan" in strategy_data:
        fueling = strategy_data["fueling_plan"]
        special_sections = fueling.get("special_sections", [])
        
        for spec in special_sections:
            km_range_str = spec.get("km_range", "")
            # Parse km range like "50-60" to get midpoint
            try:
                if "–" in km_range_str or "-" in km_range_str:
                    parts = km_range_str.replace("–", "-").split("-")
                    start = float(parts[0].strip())
                    end = float(parts[1].strip())
                    km = (start + end) / 2
                else:
                    km = float(km_range_str)
                
                # Interpolate elevation at this km
                if km >= x_km[0] and km <= x_km[-1]:
                    elev = float(np.interp(km, x_km, y))
                    marker_x.append(km)
                    marker_y.append(elev)
                    marker_labels.append("🍌 Fuel")
                    marker_types.append("fueling")
            except (ValueError, IndexError):
                continue
    
    # Mental cue markers
    if strategy_data and "mental_cues" in strategy_data:
        cues = strategy_data["mental_cues"]
        
        for cue in cues:
            try:
                km = float(cue.get("km", 0))
                
                if km >= x_km[0] and km <= x_km[-1]:
                    elev = float(np.interp(km, x_km, y))
                    marker_x.append(km)
                    marker_y.append(elev)
                    cue_text = str(cue.get("cue", ""))
                    # Truncate long cues
                    if len(cue_text) > 30:
                        cue_text = cue_text[:27] + "..."
                    marker_labels.append(f"💭 {cue_text}")
                    marker_types.append("mental_cue")
            except (ValueError, TypeError):
                continue
    
    return marker_x, marker_y, marker_labels, marker_types


def add_critical_sections_to_chart(
    fig: Any,
    strategy_data: Dict[str, Any],
    df_gpx: pd.DataFrame
) -> Any:
    """
    Add vertical spans to highlight critical sections on the chart.
    
    Args:
        fig: Plotly figure object
        strategy_data: Strategy data with critical_sections
        df_gpx: GPX DataFrame
    
    Returns:
        Modified figure
    """
    if not strategy_data or "critical_sections" not in strategy_data:
        return fig
    
    for section in strategy_data["critical_sections"]:
        start_km = float(section.get("start_km", 0))
        end_km = float(section.get("end_km", 0))
        label = str(section.get("label", ""))
        
        # Add subtle background highlight for critical sections
        fig.add_vrect(
            x0=start_km,
            x1=end_km,
            fillcolor="rgba(255, 0, 0, 0.08)",
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="rgba(255, 0, 0, 0.6)",
        )
    
    return fig

