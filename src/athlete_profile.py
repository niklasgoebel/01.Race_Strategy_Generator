# src/athlete_profile.py

def get_default_athlete_profile():
    """
    Hard-coded athlete profile for Niklas, v0.

    Later we can load this from a config file or UI inputs.
    """
    return {
        "name": "Niklas",
        "target_race": "Chianti Ultra Trail 75 km",
        "race_date": "2026-03-21",
        "experience": "intermediate-advanced road runner, newer to long mountain ultras",
        "weekly_volume_km": 90,
        "long_run_km": 30,
        "vo2max": 57,
        "max_hr": 202,
        "lactate_threshold_hr": 185,
        "lactate_threshold_pace_per_km": "4:05",
        "recent_best_5k": "18:40",
        "recent_best_half_marathon": "1:28:00",

        # Pacing preferences / constraints
        "preferred_ascent_effort": "steady, under control, avoid red-lining early",
        "descent_style": "decently confident but cautious on very steep/technical",
        "heat_tolerance": "average",
        "goal_type": "finish strong with smart pacing, not all-out racing",

        # Fueling preferences (we can refine later)
        "fuel_type": "gels + sports drink",
        "carbs_per_hour_target_g": 70,
        "hydration_notes": "will carry 2 soft flasks, refill at aid stations",
    }
