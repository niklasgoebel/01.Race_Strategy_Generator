# src/time_estimator.py
"""
Time estimation and pacing calculations for ultra races.
Uses athlete profile and course characteristics to predict finish times and splits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def vo2max_to_flat_pace_min_per_km(vo2max: int) -> float:
    """
    Estimate flat-ground pace from VO2max.
    Based on typical lactate threshold pace being ~83-88% of VO2max.
    
    Args:
        vo2max: Athlete's VO2max value
    
    Returns:
        Estimated comfortable aerobic pace in minutes per km
    """
    # Rough estimation: higher VO2max = faster pace
    # These are conservative estimates for ultra pacing (not race pace)
    if vo2max >= 65:
        return 4.0  # ~4:00/km comfortable
    elif vo2max >= 60:
        return 4.25
    elif vo2max >= 55:
        return 4.5
    elif vo2max >= 50:
        return 4.75
    elif vo2max >= 45:
        return 5.0
    else:
        return 5.5


def gradient_adjustment_factor(gradient_pct: float, is_descent: bool = False) -> float:
    """
    Calculate pace adjustment factor based on gradient.
    
    Based on research showing:
    - Uphill: ~30-50% slower per 1% gradient above 3%
    - Downhill: can be faster on moderate grades, slower on steep technical descents
    
    Args:
        gradient_pct: Average gradient percentage
        is_descent: Whether this is a descent segment
    
    Returns:
        Multiplier for pace (>1 means slower, <1 means faster)
    """
    if is_descent:
        # Descents: can gain time on moderate grades
        abs_grade = abs(gradient_pct)
        if abs_grade < 5:
            return 0.85  # 15% faster
        elif abs_grade < 10:
            return 0.9   # 10% faster
        elif abs_grade < 15:
            return 1.0   # Same pace (technical)
        else:
            return 1.2   # 20% slower (very steep/technical)
    else:
        # Climbs: progressively slower with steeper grades
        if gradient_pct < 3:
            return 1.0
        elif gradient_pct < 8:
            return 1.4  # 40% slower (power hiking)
        elif gradient_pct < 15:
            return 1.8  # 80% slower (steep hiking)
        elif gradient_pct < 25:
            return 2.3  # 130% slower (very steep)
        else:
            return 3.0  # 200% slower (extreme grades)


def experience_adjustment(experience_level: str) -> float:
    """
    Adjust estimates based on experience level.
    
    Args:
        experience_level: Description of athlete's experience
    
    Returns:
        Adjustment factor (>1 means slower/more conservative)
    """
    experience_lower = experience_level.lower()
    
    if "first" in experience_lower or "newer" in experience_lower or "beginner" in experience_lower:
        return 1.15  # Add 15% conservative buffer
    elif "experienced" in experience_lower or "advanced" in experience_lower or "multiple" in experience_lower:
        return 1.0   # Use base estimates
    elif "elite" in experience_lower or "competitive" in experience_lower:
        return 0.92  # 8% faster (strong athletes)
    else:
        return 1.05  # Default: slightly conservative


def estimate_segment_times(
    segments_df: pd.DataFrame,
    athlete_profile: Dict[str, Any]
) -> pd.DataFrame:
    """
    Estimate time for each course segment based on athlete profile.
    
    Args:
        segments_df: DataFrame with course segments (from course model)
        athlete_profile: Athlete profile dictionary
    
    Returns:
        DataFrame with additional columns: estimated_time_min, pace_min_per_km
    """
    segments = segments_df.copy()
    
    # Get base pace from VO2max
    vo2max = int(athlete_profile.get("vo2max", 50))
    base_pace = vo2max_to_flat_pace_min_per_km(vo2max)
    
    # Experience adjustment
    experience = str(athlete_profile.get("experience", ""))
    exp_factor = experience_adjustment(experience)
    
    # Calculate time for each segment
    times = []
    paces = []
    
    for _, seg in segments.iterrows():
        distance_km = float(seg["distance_km"])
        segment_type = str(seg["type"])
        gradient = float(seg["avg_gradient"])
        
        # Apply gradient adjustment
        if segment_type == "climb":
            grad_factor = gradient_adjustment_factor(abs(gradient), is_descent=False)
        elif segment_type == "descent":
            grad_factor = gradient_adjustment_factor(abs(gradient), is_descent=True)
        else:
            # Runnable/flat
            grad_factor = 1.0
        
        # Calculate adjusted pace
        adjusted_pace = base_pace * grad_factor * exp_factor
        
        # Calculate time
        time_min = distance_km * adjusted_pace
        
        times.append(time_min)
        paces.append(adjusted_pace)
    
    segments["estimated_time_min"] = times
    segments["pace_min_per_km"] = paces
    
    return segments


def calculate_split_times(
    segments_with_times: pd.DataFrame,
    aid_stations: List[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calculate cumulative split times at aid stations.
    
    Args:
        segments_with_times: DataFrame with segments and estimated times
        aid_stations: List of aid station dicts with 'name' and 'km' keys
    
    Returns:
        DataFrame with aid station splits
    """
    if not aid_stations:
        return pd.DataFrame()
    
    # Calculate cumulative time at each point
    segments = segments_with_times.copy()
    segments["cumulative_time_min"] = segments["estimated_time_min"].cumsum()
    segments["cumulative_distance_km"] = segments["end_km"]
    
    splits = []
    
    for station in aid_stations:
        station_km = float(station["km"])
        station_name = str(station["name"])
        
        # Find the segment containing this aid station
        matching_segments = segments[
            (segments["start_km"] <= station_km) & 
            (segments["end_km"] >= station_km)
        ]
        
        if matching_segments.empty:
            continue
        
        seg = matching_segments.iloc[0]
        
        # Interpolate time within the segment
        seg_start_km = float(seg["start_km"])
        seg_end_km = float(seg["end_km"])
        seg_start_time = float(seg["cumulative_time_min"]) - float(seg["estimated_time_min"])
        seg_end_time = float(seg["cumulative_time_min"])
        
        # Linear interpolation
        if seg_end_km > seg_start_km:
            fraction = (station_km - seg_start_km) / (seg_end_km - seg_start_km)
            station_time = seg_start_time + fraction * (seg_end_time - seg_start_time)
        else:
            station_time = seg_start_time
        
        splits.append({
            "name": station_name,
            "km": station_km,
            "cumulative_time_min": station_time,
            "cumulative_time_str": format_time_hhmm(station_time),
        })
    
    return pd.DataFrame(splits)


def calculate_finish_time_range(
    segments_with_times: pd.DataFrame,
    confidence_range: float = 0.15
) -> Tuple[float, float, float]:
    """
    Calculate finish time range (conservative, expected, aggressive).
    
    Args:
        segments_with_times: DataFrame with estimated segment times
        confidence_range: Percentage range for conservative/aggressive estimates
    
    Returns:
        Tuple of (conservative_min, expected_min, aggressive_min)
    """
    total_time = float(segments_with_times["estimated_time_min"].sum())
    
    conservative = total_time * (1 + confidence_range)
    expected = total_time
    aggressive = total_time * (1 - confidence_range)
    
    return conservative, expected, aggressive


def format_time_hhmm(minutes: float) -> str:
    """
    Format time in minutes to HH:MM string.
    
    Args:
        minutes: Time in minutes
    
    Returns:
        Formatted string like "5:23" or "12:45"
    """
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"


def categorize_time_of_day(time_str: str) -> str:
    """
    Categorize time of day for strategy purposes.
    
    Args:
        time_str: Time in "HH:MM" format
    
    Returns:
        Category: early_morning, late_morning, afternoon, evening, or night
    """
    try:
        hour = int(time_str.split(":")[0])
    except (ValueError, IndexError):
        return "unknown"
    
    if 5 <= hour < 9:
        return "early_morning"  # Cool, fresh
    elif 9 <= hour < 12:
        return "late_morning"  # Warming up
    elif 12 <= hour < 16:
        return "afternoon"  # Hot, sun exposure
    elif 16 <= hour < 20:
        return "evening"  # Cooling down
    else:
        return "night"  # Headlamp, cold


def add_time_of_day_estimates(
    segments_df: pd.DataFrame,
    athlete_profile: Dict[str, Any],
    race_start_time: str = "07:00"
) -> pd.DataFrame:
    """
    Add time-of-day estimates to segments based on pacing model.
    
    Args:
        segments_df: Course segments DataFrame
        athlete_profile: Athlete profile for pace estimation
        race_start_time: Race start time in "HH:MM" format (24-hour)
    
    Returns:
        DataFrame with added columns:
        - estimated_time_min: Time for this segment
        - cumulative_time_min: Total time to end of segment
        - estimated_time_of_day: Time-of-day at segment end
        - time_of_day_category: Category (early_morning, afternoon, etc.)
    """
    from datetime import datetime, timedelta
    
    # Get time estimates for each segment
    segments_with_times = estimate_segment_times(segments_df, athlete_profile)
    
    # Calculate cumulative time
    segments_with_times["cumulative_time_min"] = segments_with_times["estimated_time_min"].cumsum()
    
    # Parse race start time
    try:
        start_dt = datetime.strptime(race_start_time, "%H:%M")
    except ValueError:
        # Default to 7am if invalid format
        start_dt = datetime.strptime("07:00", "%H:%M")
    
    # Calculate time-of-day for each segment
    def calc_time_of_day(cumulative_min):
        segment_dt = start_dt + timedelta(minutes=float(cumulative_min))
        return segment_dt.strftime("%H:%M")
    
    segments_with_times["estimated_time_of_day"] = segments_with_times["cumulative_time_min"].apply(
        calc_time_of_day
    )
    
    segments_with_times["time_of_day_category"] = segments_with_times["estimated_time_of_day"].apply(
        categorize_time_of_day
    )
    
    return segments_with_times


def reverse_calculate_required_pace(
    target_finish_time_min: float,
    segments_df: pd.DataFrame,
    athlete_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Given a target finish time, calculate required effort level.
    
    Args:
        target_finish_time_min: Desired finish time in minutes
        segments_df: Course segments DataFrame
        athlete_profile: Athlete profile
    
    Returns:
        Dictionary with required pace info and feasibility assessment
    """
    # First get baseline estimate
    segments_with_times = estimate_segment_times(segments_df, athlete_profile)
    baseline_time = float(segments_with_times["estimated_time_min"].sum())
    
    # Calculate required speedup factor
    if baseline_time <= 0:
        return {"feasible": False, "message": "Invalid baseline estimate"}
    
    required_factor = baseline_time / target_finish_time_min
    
    # Assess feasibility
    if required_factor < 0.85:
        feasibility = "Very aggressive - may not be achievable"
    elif required_factor < 0.95:
        feasibility = "Aggressive - requires strong execution"
    elif required_factor < 1.05:
        feasibility = "Realistic - matches your profile"
    elif required_factor < 1.15:
        feasibility = "Conservative - comfortable pace"
    else:
        feasibility = "Very conservative"
    
    return {
        "feasible": True,
        "target_time_min": target_finish_time_min,
        "target_time_str": format_time_hhmm(target_finish_time_min),
        "baseline_time_min": baseline_time,
        "baseline_time_str": format_time_hhmm(baseline_time),
        "required_factor": required_factor,
        "feasibility": feasibility,
        "speedup_pct": (required_factor - 1.0) * 100,
    }

