# src/athlete_profile.py
"""
Athlete profile management and utilities.
Provides backwards compatibility and helper functions for profile operations.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from src.profile_manager import ProfileManager


def get_default_athlete_profile():
    """
    Returns the default athlete profile (First Ultra preset).
    Maintained for backwards compatibility with existing notebooks/tests.
    
    For new code, use ProfileManager.load_profile() instead.
    """
    # Return the First Ultra template as default
    templates = ProfileManager.get_profile_templates()
    return templates["first_ultra_50k"].copy()


def list_athlete_profiles() -> List[str]:
    """Returns list of available athlete profile names."""
    manager = ProfileManager()
    return manager.list_profiles()


def load_athlete_profile(name: str) -> Dict[str, Any]:
    """Load an athlete profile by name."""
    manager = ProfileManager()
    return manager.load_profile(name)


def save_athlete_profile(profile: Dict[str, Any], name: Optional[str] = None) -> str:
    """
    Save an athlete profile. Returns the saved profile name.
    If name is not provided, uses profile['name'].
    """
    manager = ProfileManager()
    return manager.save_profile(profile, name)


def delete_athlete_profile(name: str) -> bool:
    """Delete an athlete profile. Returns True if successful."""
    manager = ProfileManager()
    return manager.delete_profile(name)


def get_profile_templates() -> Dict[str, Dict[str, Any]]:
    """Returns preset profile templates for different athlete types."""
    return ProfileManager.get_profile_templates()


def infer_secondary_metrics(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates secondary metrics from primary data.
    Useful for smart defaults in profile creation.
    """
    return ProfileManager.infer_secondary_metrics(profile_data)


def calculate_athlete_capabilities(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive actionable capabilities from athlete profile.
    
    This converts raw profile data (VO2max, experience) into concrete
    pacing thresholds and capabilities that the LLM can use directly
    for strategy generation.
    
    Returns:
        Dictionary with:
        - flat_comfortable_pace_min_per_km: Sustainable flat pace
        - power_hike_threshold_gradient_pct: When to switch from running to hiking
        - risk_tolerance: low/medium/high based on goal
        - fueling_capacity_g_per_hour: Carb intake capacity
        - descent_comfort_level: confident/moderate/cautious
        - estimated_race_time_multiplier: Relative to flat pace (for rough estimates)
    """
    vo2max = profile.get("vo2max", 50)
    experience = profile.get("experience", "").lower()
    
    # Estimate comfortable flat pace from VO2max
    # VO2max 60+ = elite (4:00/km), 55-60 = advanced (4:20/km), 50-55 = intermediate (4:40/km), <50 = beginner (5:00+/km)
    if vo2max >= 60:
        flat_pace = 4.0
    elif vo2max >= 55:
        flat_pace = 4.3
    elif vo2max >= 50:
        flat_pace = 4.7
    else:
        flat_pace = 5.2
    
    # Determine power hiking threshold based on experience
    # Elite athletes can run steeper grades, beginners need to hike earlier
    if "elite" in experience or "advanced" in experience:
        hike_threshold_pct = 12  # Can run up to 12% grade
    elif "beginner" in experience or "first" in experience or "newer" in experience:
        hike_threshold_pct = 6   # Should hike anything over 6%
    else:
        hike_threshold_pct = 8   # Intermediate: hike over 8%
    
    # Risk tolerance from goal type
    goal = profile.get("goal_type", "").lower()
    if "conservative" in goal or "finish" in goal or "smart" in goal:
        risk_tolerance = "low"
        time_multiplier = 1.3  # Very conservative pacing
    elif "pr" in goal or "race" in goal or "fast" in goal or "aggressive" in goal:
        risk_tolerance = "high"
        time_multiplier = 1.0  # Aggressive pacing
    else:
        risk_tolerance = "medium"
        time_multiplier = 1.15  # Moderate pacing
    
    # Descent comfort from profile or infer from experience
    descent_style = profile.get("descent_style", "").lower()
    if "confident" in descent_style or "fast" in descent_style:
        descent_comfort = "confident"
    elif "cautious" in descent_style or "careful" in descent_style:
        descent_comfort = "cautious"
    else:
        descent_comfort = "moderate"
    
    return {
        "flat_comfortable_pace_min_per_km": float(flat_pace),
        "power_hike_threshold_gradient_pct": float(hike_threshold_pct),
        "risk_tolerance": risk_tolerance,
        "fueling_capacity_g_per_hour": int(profile.get("carbs_per_hour_target_g", 60)),
        "descent_comfort_level": descent_comfort,
        "estimated_race_time_multiplier": float(time_multiplier),
    }


def ensure_default_profiles_exist():
    """
    Ensures default preset profiles exist in data/profiles/.
    Creates template examples if they don't exist.
    """
    manager = ProfileManager()
    
    # Save preset profile templates
    templates = get_profile_templates()
    for template_id, template_data in templates.items():
        # Only save if doesn't already exist (avoid overwriting user edits)
        profile_name = template_data["name"]
        if not manager.profile_exists(profile_name):
            manager.save_profile(template_data, profile_name)
