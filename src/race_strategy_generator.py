# src/race_strategy_generator.py

from __future__ import annotations

import json
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .athlete_profile import get_default_athlete_profile

# Make sure environment variables from .env are loaded
load_dotenv()

# OpenAI client – uses OPENAI_API_KEY from environment
client = OpenAI()


def build_strategy_prompt(
    course_summary: Dict[str, Any],
    segment_summaries: List[str],
    climb_summaries: List[str],
    athlete_profile: Dict[str, Any],
) -> str:
    """
    Build a structured prompt for the LLM.

    We ask for TWO things:
      1) A readable race strategy in natural language.
      2) A structured JSON object following a defined schema.
    """

    max_segments = 40
    segment_text = "\n".join(segment_summaries[:max_segments])

    max_climbs = 10
    climb_text = "\n".join(climb_summaries[:max_climbs])

    # ---------------------
    # JSON SCHEMA DESIGN
    # ---------------------
    json_instruction = """
After the readable explanation, output ONLY a JSON object with EXACTLY the following structure:

{
  "global_strategy": {
    "early": "string – strategy for roughly 0–25 km",
    "mid": "string – strategy for roughly 25–50 km",
    "late": "string – strategy for roughly 50–75 km"
  },

  "critical_sections": [
    {
      "label": "string – e.g. 'Climb #1'",
      "start_km": float,
      "end_km": float,
      "gain_m": float,
      "avg_gradient_pct": float,
      "effort_rpe": "string – e.g. '4-5'",
      "effort_hr_percent_max": "string – e.g. '75-85%'",
      "notes": "string – specific tips for this section"
    }
  ],

  "pacing_chunks": [
    {
      "start_km": float,
      "end_km": float,
      "terrain_summary": "string",
      "effort_rpe": "string",
      "effort_hr_percent_max": "string",
      "key_focus": "string"
    }
  ],

  "fueling_plan": {
    "carbs_g_per_hour": int,
    "hydration_notes": "string",
    "special_sections": [
      {
        "km_range": "string – e.g. '50–60'",
        "reason": "string",
        "fueling_focus": "string"
      }
    ]
  },

  "mental_cues": [
    {
      "km": float,
      "cue": "string"
    }
  ]
}

IMPORTANT:
- The JSON MUST be valid and parsable.
- Use double quotes for all strings.
- No trailing commas.
- Do not wrap the JSON in backticks or Markdown.
"""

    prompt = f"""
You are an experienced ultra-trail running coach and race strategist.

You are helping an athlete plan their race strategy for the following event:

RACE
- Name: {course_summary.get("race_name", "Chianti Ultra Trail 75 km")}
- Distance: {course_summary.get("total_distance_km")} km
- Total elevation gain: {course_summary.get("total_gain_m")} m
- Total elevation loss: {course_summary.get("total_loss_m")} m
- Number of segments (modelled): {course_summary.get("num_segments")}

ATHLETE PROFILE
- Name: {athlete_profile.get("name")}
- Experience: {athlete_profile.get("experience")}
- Weekly volume: ~{athlete_profile.get("weekly_volume_km")} km
- Long run: ~{athlete_profile.get("long_run_km")} km
- VO2max: {athlete_profile.get("vo2max")}
- Max HR: {athlete_profile.get("max_hr")}
- Lactate threshold HR: {athlete_profile.get("lactate_threshold_hr")}
- Lactate threshold pace: {athlete_profile.get("lactate_threshold_pace_per_km")} min/km
- Goal type: {athlete_profile.get("goal_type")}
- Fuel type: {athlete_profile.get("fuel_type")}
- Target carbs per hour: {athlete_profile.get("carbs_per_hour_target_g")} g

NOTES:
- Max HR: {athlete_profile.get("max_hr")}
- Lactate threshold HR: {athlete_profile.get("lactate_threshold_hr")}
- Goal type: {athlete_profile.get("goal_type")}
- Fueling target: {athlete_profile.get("carbs_per_hour_target_g")} g carbs/hour

COURSE OVERVIEW (SEGMENTS)
Each row:
  type | start_km–end_km | distance_km | gain/loss | avg_gradient
{segment_text}

KEY CLIMBS (MOST IMPORTANT FEATURES)
{climb_text}

TASK

You are NOT writing general training advice. You are writing a CONCRETE RACE STRATEGY for this one event for this specific athlete.

Please:

1) First, write a clear, concise, human-readable race strategy covering:
   - how to approach the early / mid / late parts of the race,
   - where to be patient vs where to be brave,
   - how to tackle the major climbs and technical descents,
   - how to think about effort (RPE and % of max HR),
   - basic fueling/hydration guidance.

2) Then, on a new line, output ONLY a JSON object that follows the schema below exactly.

{json_instruction}
"""
    return prompt


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to locate and parse a JSON object inside a larger text response.

    Handles cases where the model wraps JSON in ```json ... ``` fences.
    """
    cleaned = text.strip()

    # Handle ```json ... ``` style wrapping
    if "```" in cleaned:
        first_fence = cleaned.find("```")
        last_fence = cleaned.rfind("```")
        inside = cleaned[first_fence + 3 : last_fence].strip()
        if inside.lower().startswith("json"):
            inside = inside[4:].strip()
        cleaned = inside

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None

    json_str = cleaned[first_brace : last_brace + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def generate_race_strategy(
    course_summary: Dict[str, Any],
    segment_summaries: List[str],
    climb_summaries: List[str],
    athlete_profile: Optional[Dict[str, Any]] = None,
    model: str = "gpt-5-mini",
    max_output_tokens: int = 1800,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Generate a race strategy using the OpenAI Responses API.

    Returns:
        (human_readable_text, structured_json_or_none)
    """
    if athlete_profile is None:
        athlete_profile = get_default_athlete_profile()

    prompt = build_strategy_prompt(
        course_summary=course_summary,
        segment_summaries=segment_summaries,
        climb_summaries=climb_summaries,
        athlete_profile=athlete_profile,
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_output_tokens=max_output_tokens,
    )

    # Extract raw text from the response (Responses API format)
    raw_text = response.output[0].content[0].text

    # Split: human narrative + JSON (if possible)
    parsed_json = _extract_json_from_text(raw_text)

    return raw_text, parsed_json


def main_demo():
    """
    Simple CLI-style demo stub.

    This file is meant to be imported from a notebook or another script
    that already has:
      - course_summary
      - segment_summaries
      - climb_summaries

    Example (from a notebook):

        from src.race_strategy_generator import generate_race_strategy
        from src.athlete_profile import get_default_athlete_profile

        athlete = get_default_athlete_profile()
        text, data = generate_race_strategy(
            course_summary,
            segment_summaries,
            climb_summaries,
            athlete_profile=athlete,
        )

    """
    print("race_strategy_generator.py is intended to be imported, not run directly.")
    print("Use generate_race_strategy(course_summary, segment_summaries, climb_summaries).")


if __name__ == "__main__":
    main_demo()