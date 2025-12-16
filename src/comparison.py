# src/comparison.py
"""
Strategy comparison functionality for analyzing different athlete profiles or scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def compare_strategies(result1: Any, result2: Any) -> Dict[str, Any]:
    """
    Compare two race strategy results and identify differences.
    
    Args:
        result1: First PipelineResult
        result2: Second PipelineResult
    
    Returns:
        Dictionary with comparison data and differences
    """
    comparison = {
        "course_differences": _compare_course_summaries(
            result1.course_summary, result2.course_summary
        ),
        "time_differences": _compare_time_estimates(result1, result2),
        "strategy_differences": _compare_strategy_data(
            result1.strategy_data, result2.strategy_data
        ),
        "critical_sections_diff": _compare_critical_sections(result1, result2),
    }
    
    return comparison


def _compare_course_summaries(
    course1: Dict[str, Any], course2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare course summary metrics."""
    # Course should be the same, but check anyway
    return {
        "same_course": (
            course1.get("total_distance_km") == course2.get("total_distance_km") and
            course1.get("total_gain_m") == course2.get("total_gain_m")
        ),
        "distance_km": course1.get("total_distance_km", 0),
        "gain_m": course1.get("total_gain_m", 0),
    }


def _compare_time_estimates(result1: Any, result2: Any) -> Dict[str, Any]:
    """Compare finish time estimates."""
    if not result1.finish_time_range or not result2.finish_time_range:
        return {"available": False}
    
    time1 = result1.finish_time_range["expected_min"]
    time2 = result2.finish_time_range["expected_min"]
    
    diff_min = time2 - time1
    diff_pct = (diff_min / time1) * 100 if time1 > 0 else 0
    
    return {
        "available": True,
        "time1_min": time1,
        "time1_str": result1.finish_time_range["expected_str"],
        "time2_min": time2,
        "time2_str": result2.finish_time_range["expected_str"],
        "difference_min": diff_min,
        "difference_pct": diff_pct,
        "faster": "Profile 1" if diff_min < 0 else "Profile 2" if diff_min > 0 else "Same",
    }


def _compare_strategy_data(
    strategy1: Dict[str, Any], strategy2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare overall strategy recommendations."""
    if not strategy1 or not strategy2:
        return {"available": False}
    
    # Compare fueling plans
    fueling_diff = {}
    if "fueling_plan" in strategy1 and "fueling_plan" in strategy2:
        carbs1 = strategy1["fueling_plan"].get("carbs_g_per_hour", 0)
        carbs2 = strategy2["fueling_plan"].get("carbs_g_per_hour", 0)
        
        fueling_diff = {
            "carbs1": carbs1,
            "carbs2": carbs2,
            "difference": carbs2 - carbs1,
        }
    
    # Compare number of pacing chunks
    chunks1 = len(strategy1.get("pacing_chunks", []))
    chunks2 = len(strategy2.get("pacing_chunks", []))
    
    return {
        "available": True,
        "fueling": fueling_diff,
        "pacing_chunks_count": {"profile1": chunks1, "profile2": chunks2},
    }


def _compare_critical_sections(result1: Any, result2: Any) -> List[Dict[str, Any]]:
    """Compare critical section recommendations."""
    if (not result1.strategy_data or "critical_sections" not in result1.strategy_data or
        not result2.strategy_data or "critical_sections" not in result2.strategy_data):
        return []
    
    sections1 = result1.strategy_data["critical_sections"]
    sections2 = result2.strategy_data["critical_sections"]
    
    comparisons = []
    
    # Match sections by km range (they should be the same sections)
    for s1 in sections1:
        start1 = s1.get("start_km", 0)
        end1 = s1.get("end_km", 0)
        
        # Find corresponding section in strategy 2
        matched = None
        for s2 in sections2:
            start2 = s2.get("start_km", 0)
            end2 = s2.get("end_km", 0)
            
            # Allow small tolerance for matching
            if abs(start2 - start1) < 0.5 and abs(end2 - end1) < 0.5:
                matched = s2
                break
        
        if matched:
            comparison = {
                "label": s1.get("label", "Section"),
                "km_range": f"{start1:.1f}-{end1:.1f}",
                "effort1": s1.get("effort_rpe", "-"),
                "effort2": matched.get("effort_rpe", "-"),
                "hr1": s1.get("effort_hr_percent_max", "-"),
                "hr2": matched.get("effort_hr_percent_max", "-"),
                "notes1": s1.get("notes", ""),
                "notes2": matched.get("notes", ""),
                "different": (
                    s1.get("effort_rpe") != matched.get("effort_rpe") or
                    s1.get("notes") != matched.get("notes")
                ),
            }
            comparisons.append(comparison)
    
    return comparisons


def highlight_differences(
    strategy1: Dict[str, Any], strategy2: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Identify and format key differences between two strategies.
    
    Args:
        strategy1: First strategy data dictionary
        strategy2: Second strategy data dictionary
    
    Returns:
        Dictionary with categories of differences
    """
    differences = {
        "pacing": [],
        "fueling": [],
        "mental_approach": [],
    }
    
    # Pacing differences
    if "global_strategy" in strategy1 and "global_strategy" in strategy2:
        gs1 = strategy1["global_strategy"]
        gs2 = strategy2["global_strategy"]
        
        for phase in ["early", "mid", "late"]:
            if gs1.get(phase) != gs2.get(phase):
                differences["pacing"].append(
                    f"{phase.capitalize()} phase: Different approach"
                )
    
    # Fueling differences
    if "fueling_plan" in strategy1 and "fueling_plan" in strategy2:
        fp1 = strategy1["fueling_plan"]
        fp2 = strategy2["fueling_plan"]
        
        carbs1 = fp1.get("carbs_g_per_hour", 0)
        carbs2 = fp2.get("carbs_g_per_hour", 0)
        
        if carbs1 != carbs2:
            diff = carbs2 - carbs1
            differences["fueling"].append(
                f"Carb intake differs by {abs(diff)}g/hour "
                f"({'higher' if diff > 0 else 'lower'} in Profile 2)"
            )
    
    # Mental cue count differences
    cues1 = len(strategy1.get("mental_cues", []))
    cues2 = len(strategy2.get("mental_cues", []))
    
    if cues1 != cues2:
        differences["mental_approach"].append(
            f"Different number of mental cues: {cues1} vs {cues2}"
        )
    
    return differences


def create_comparison_dataframe(
    result1: Any, result2: Any, profile1_name: str, profile2_name: str
) -> pd.DataFrame:
    """
    Create a DataFrame comparing key metrics between two strategies.
    
    Args:
        result1: First PipelineResult
        result2: Second PipelineResult
        profile1_name: Name of first profile
        profile2_name: Name of second profile
    
    Returns:
        DataFrame with comparison metrics
    """
    metrics = []
    
    # Finish time comparison
    if result1.finish_time_range and result2.finish_time_range:
        metrics.append({
            "Metric": "Expected Finish Time",
            profile1_name: result1.finish_time_range["expected_str"],
            profile2_name: result2.finish_time_range["expected_str"],
        })
        
        metrics.append({
            "Metric": "Conservative Time",
            profile1_name: result1.finish_time_range["conservative_str"],
            profile2_name: result2.finish_time_range["conservative_str"],
        })
        
        metrics.append({
            "Metric": "Aggressive Time",
            profile1_name: result1.finish_time_range["aggressive_str"],
            profile2_name: result2.finish_time_range["aggressive_str"],
        })
    
    # Fueling comparison
    if (result1.strategy_data and "fueling_plan" in result1.strategy_data and
        result2.strategy_data and "fueling_plan" in result2.strategy_data):
        
        carbs1 = result1.strategy_data["fueling_plan"].get("carbs_g_per_hour", "-")
        carbs2 = result2.strategy_data["fueling_plan"].get("carbs_g_per_hour", "-")
        
        metrics.append({
            "Metric": "Carbs (g/hour)",
            profile1_name: f"{carbs1}",
            profile2_name: f"{carbs2}",
        })
    
    # Strategy complexity (number of critical sections)
    if (result1.strategy_data and "critical_sections" in result1.strategy_data and
        result2.strategy_data and "critical_sections" in result2.strategy_data):
        
        sections1 = len(result1.strategy_data["critical_sections"])
        sections2 = len(result2.strategy_data["critical_sections"])
        
        metrics.append({
            "Metric": "Critical Sections",
            profile1_name: f"{sections1}",
            profile2_name: f"{sections2}",
        })
    
    return pd.DataFrame(metrics)

