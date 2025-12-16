# src/profile_manager.py
"""
Athlete profile management: CRUD operations, validation, and template generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PROFILES_DIR = Path("data/profiles")


class ProfileManager:
    """Manages athlete profiles with storage, validation, and templates."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[str]:
        """Returns list of available profile names (without .json extension)."""
        if not self.profiles_dir.exists():
            return []
        
        profiles = []
        for p in self.profiles_dir.glob("*.json"):
            profiles.append(p.stem)
        
        return sorted(profiles)

    def load_profile(self, name: str) -> Dict[str, Any]:
        """Load a profile by name. Raises FileNotFoundError if not found."""
        profile_path = self.profiles_dir / f"{name}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found at {profile_path}")
        
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_profile(self, profile: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        Save a profile. If name is not provided, uses profile['name'].
        Returns the saved profile name.
        """
        profile_name = name or profile.get("name")
        
        if not profile_name:
            raise ValueError("Profile must have a 'name' field or name parameter must be provided")
        
        # Validate profile has required fields
        self._validate_profile(profile)
        
        # Ensure name is in the profile dict
        profile["name"] = profile_name
        
        profile_path = self.profiles_dir / f"{profile_name}.json"
        
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        
        return profile_name

    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name. Returns True if deleted, False if not found."""
        profile_path = self.profiles_dir / f"{name}.json"
        
        if not profile_path.exists():
            return False
        
        profile_path.unlink()
        return True

    def profile_exists(self, name: str) -> bool:
        """Check if a profile exists."""
        profile_path = self.profiles_dir / f"{name}.json"
        return profile_path.exists()

    def _validate_profile(self, profile: Dict[str, Any]) -> None:
        """
        Validate that profile has required fields.
        
        Required fields: name, experience, weekly_volume_km, long_run_km, goal_type, fuel_type
        Optional fields (will be auto-inferred if missing): vo2max, max_hr, lactate_threshold_hr, 
        lactate_threshold_pace_per_km, carbs_per_hour_target_g
        """
        required_fields = [
            "name",
            "experience",
            "weekly_volume_km",
            "long_run_km",
            "goal_type",
            "fuel_type",
        ]
        
        missing = [f for f in required_fields if f not in profile]
        
        if missing:
            raise ValueError(f"Profile missing required fields: {', '.join(missing)}")

    @staticmethod
    def get_profile_templates() -> Dict[str, Dict[str, Any]]:
        """
        Returns 3 preset profile templates for different athlete types.
        Uses standardized dropdown values for experience, goal_type, and fuel_type.
        """
        return {
            "first_ultra_50k": {
                "name": "First Ultra",
                "target_race": "50k ultra",
                "race_date": "",
                "experience": "Beginner/First Ultra",
                "weekly_volume_km": 60,
                "long_run_km": 25,
                "vo2max": 50,
                "max_hr": 190,
                "lactate_threshold_hr": 172,
                "lactate_threshold_pace_per_km": "4:45",
                "recent_best_5k": "21:00",
                "recent_best_half_marathon": "1:40:00",
                "preferred_ascent_effort": "steady and controlled, avoid going too hard early",
                "descent_style": "cautious, focus on staying safe",
                "heat_tolerance": "average",
                "goal_type": "Finish comfortably",
                "fuel_type": "Gels + sports drink",
                "carbs_per_hour_target_g": 60,
                "hydration_notes": "carry handheld or vest, refill at aid stations",
            },
            "experienced_ultra_100k": {
                "name": "Experienced Ultra Runner (100k+)",
                "target_race": "100k+ ultra",
                "race_date": "",
                "experience": "Experienced (multiple ultras)",
                "weekly_volume_km": 90,
                "long_run_km": 35,
                "vo2max": 55,
                "max_hr": 188,
                "lactate_threshold_hr": 173,
                "lactate_threshold_pace_per_km": "4:15",
                "recent_best_5k": "19:30",
                "recent_best_half_marathon": "1:32:00",
                "preferred_ascent_effort": "steady, push on longer climbs but stay aerobic",
                "descent_style": "confident and efficient, can make up time",
                "heat_tolerance": "good, experienced with heat management",
                "goal_type": "Strong performance",
                "fuel_type": "Mixed (gels, bars, real food)",
                "carbs_per_hour_target_g": 75,
                "hydration_notes": "soft flasks + vest, experienced with hydration strategy",
            },
            "road_to_trail_transition": {
                "name": "Road Runner → Trail Transition",
                "target_race": "first trail ultra",
                "race_date": "",
                "experience": "Intermediate (some ultras)",
                "weekly_volume_km": 75,
                "long_run_km": 28,
                "vo2max": 54,
                "max_hr": 192,
                "lactate_threshold_hr": 178,
                "lactate_threshold_pace_per_km": "4:20",
                "recent_best_5k": "19:00",
                "recent_best_half_marathon": "1:30:00",
                "preferred_ascent_effort": "learning to pace climbs, tendency to go too hard",
                "descent_style": "still developing confidence on technical descents",
                "heat_tolerance": "good, from road racing experience",
                "goal_type": "Smart pacing, finish strong",
                "fuel_type": "Gels + sports drink",
                "carbs_per_hour_target_g": 70,
                "hydration_notes": "vest with soft flasks, learning ultra hydration needs",
            },
        }

    @staticmethod
    def infer_secondary_metrics(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates secondary metrics from primary data where possible.
        Returns updated profile dict with inferred values.
        
        Enhanced to handle more missing fields and infer from experience level + training volume.
        """
        profile = profile_data.copy()
        
        # Get experience level for intelligent defaults
        experience = profile.get("experience", "").lower()
        weekly_volume = profile.get("weekly_volume_km", 60)
        
        # Estimate VO2max from experience + weekly volume if not provided
        if "vo2max" not in profile or profile.get("vo2max", 0) == 0:
            # Try from recent 5K time first
            if "recent_best_5k" in profile and profile.get("recent_best_5k"):
                time_str = str(profile["recent_best_5k"])
                try:
                    if ":" in time_str:
                        parts = time_str.split(":")
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        total_seconds = minutes * 60 + seconds
                        
                        # Rough VO2max estimation from 5K time
                        # Formula: VO2max ≈ -4.6 + 0.182258 × velocity(m/min) + 0.000104 × velocity²
                        velocity_m_per_min = 5000 / (total_seconds / 60)
                        vo2max = -4.6 + 0.182258 * velocity_m_per_min + 0.000104 * velocity_m_per_min ** 2
                        profile["vo2max"] = int(vo2max)
                except (ValueError, IndexError):
                    pass
            
            # If still not set, estimate from experience + volume
            if "vo2max" not in profile or profile.get("vo2max", 0) == 0:
                # Base VO2max from experience level
                if "advanced" in experience or "elite" in experience:
                    base_vo2max = 58
                elif "experienced" in experience or "multiple" in experience:
                    base_vo2max = 54
                elif "intermediate" in experience:
                    base_vo2max = 52
                else:  # beginner/first ultra
                    base_vo2max = 48
                
                # Adjust based on weekly volume (+/- 3 points for volume variance)
                if weekly_volume >= 90:
                    base_vo2max += 3
                elif weekly_volume >= 70:
                    base_vo2max += 1
                elif weekly_volume < 50:
                    base_vo2max -= 2
                
                profile["vo2max"] = base_vo2max
        
        # Estimate max HR from age if provided, otherwise use experience-based default
        if "max_hr" not in profile or profile.get("max_hr", 0) == 0:
            if "age" in profile and profile.get("age"):
                age = int(profile["age"])
                profile["max_hr"] = 220 - age
            else:
                # Experience-based age assumption for max HR
                if "advanced" in experience or "elite" in experience:
                    profile["max_hr"] = 185  # Assumes ~35 years old
                elif "experienced" in experience:
                    profile["max_hr"] = 188  # Assumes ~32 years old
                elif "intermediate" in experience:
                    profile["max_hr"] = 190  # Assumes ~30 years old
                else:  # beginner
                    profile["max_hr"] = 192  # Assumes ~28 years old
        
        # Estimate lactate threshold HR (typically 88-92% of max HR based on experience)
        if "lactate_threshold_hr" not in profile or profile.get("lactate_threshold_hr", 0) == 0:
            max_hr = int(profile.get("max_hr", 190))
            # More experienced athletes can sustain higher % of max HR
            if "advanced" in experience or "elite" in experience:
                lt_percent = 0.92
            elif "experienced" in experience:
                lt_percent = 0.91
            elif "intermediate" in experience:
                lt_percent = 0.90
            else:  # beginner
                lt_percent = 0.88
            
            profile["lactate_threshold_hr"] = int(max_hr * lt_percent)
        
        # Estimate threshold pace from VO2max
        if "lactate_threshold_pace_per_km" not in profile or not profile.get("lactate_threshold_pace_per_km"):
            vo2max = int(profile.get("vo2max", 50))
            # Rough estimate: threshold pace in seconds per km
            if vo2max >= 60:
                pace_sec = 240  # 4:00/km
            elif vo2max >= 56:
                pace_sec = 250  # 4:10/km
            elif vo2max >= 54:
                pace_sec = 260  # 4:20/km
            elif vo2max >= 52:
                pace_sec = 270  # 4:30/km
            elif vo2max >= 50:
                pace_sec = 280  # 4:40/km
            else:
                pace_sec = 290  # 4:50/km
            
            minutes = pace_sec // 60
            seconds = pace_sec % 60
            profile["lactate_threshold_pace_per_km"] = f"{minutes}:{seconds:02d}"
        
        # Estimate carbs per hour based on experience (LLM will override this based on actual strategy)
        if "carbs_per_hour_target_g" not in profile:
            if "advanced" in experience or "elite" in experience:
                profile["carbs_per_hour_target_g"] = 80  # Trained gut
            elif "experienced" in experience:
                profile["carbs_per_hour_target_g"] = 75
            elif "intermediate" in experience:
                profile["carbs_per_hour_target_g"] = 70
            else:  # beginner
                profile["carbs_per_hour_target_g"] = 60  # Conservative
        
        return profile


def get_default_profile_manager() -> ProfileManager:
    """Returns a ProfileManager instance with default settings."""
    return ProfileManager()

