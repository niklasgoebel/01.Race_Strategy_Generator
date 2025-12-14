from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

# (Optional) useful for demos/tests
from .athlete_profile import get_default_athlete_profile  # noqa: F401

load_dotenv()
client = OpenAI()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_strategy_prompt(
    course_summary: Dict[str, Any],
    segment_summaries: List[str],
    climb_summaries: List[str],
    climb_block_summaries: List[str],
    athlete_profile: Dict[str, Any],
    *,
    json_only: bool = False,
) -> str:
    """
    Build a structured prompt for the LLM.

    If json_only=True, instruct the model to output ONLY valid JSON.
    Otherwise, ask for coach notes (human) then JSON (machine).
    """
    max_segments = 40
    segment_text = "\n".join(segment_summaries[:max_segments]) if segment_summaries else "—"

    max_climbs = 10
    climb_text = "\n".join(climb_summaries[:max_climbs]) if climb_summaries else "—"

    max_blocks = 10
    climb_blocks_text = "\n".join(climb_block_summaries[:max_blocks]) if climb_block_summaries else "—"

    json_schema = """
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
""".strip()

    common_context = f"""
You are an experienced ultra-trail running coach and race strategist.

RACE
- Name: {course_summary.get("race_name", "Race")}
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

COURSE OVERVIEW (SEGMENTS)
Each row:
  type | start_km–end_km | distance_km | gain/loss | avg_gradient
{segment_text}

KEY CLIMBS (MOST IMPORTANT FEATURES)
{climb_text}

CLIMB BLOCKS (SUSTAINED EFFORT STRUCTURE)
These represent sustained climbing efforts that may include SHORT runnable interruptions.
IMPORTANT: Treat these as continuous “costly” effort with LIMITED recovery.

{climb_blocks_text}
""".strip()

    if json_only:
        task_block = f"""
TASK
You MUST output ONLY a valid JSON object following the schema below.
- No markdown fences. No extra commentary. Only JSON.
- Use double quotes for all strings.
- No trailing commas.

Return EXACTLY this schema (same keys, same nesting, same field types):
{json_schema}
""".strip()
    else:
        task_block = f"""
TASK
You MUST output in TWO PARTS, in this exact order:

PART 1 — COACH NOTES (human-readable)
- Write 6–12 bullet points.
- Short, concrete, actionable.
- Reference specific course sections (km ranges, climbs, climb blocks).
- Include pacing guidance using RPE + % of max HR.
- Include fueling reminders where relevant.
- Do NOT include any JSON, braces, or code blocks in Part 1.

PART 2 — JSON (machine-readable)
- On a new line after Part 1, output ONLY a valid JSON object following the schema below.
- No markdown fences. No extra commentary. Only JSON.
- Use double quotes for all strings.
- No trailing commas.

JSON schema:
{json_schema}
""".strip()

    return f"{common_context}\n\n{task_block}".strip()


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()

    # Handle ```json ... ``` style wrapping
    if "```" in cleaned:
        first_fence = cleaned.find("```")
        last_fence = cleaned.rfind("```")
        if first_fence != -1 and last_fence != -1 and last_fence > first_fence:
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


def _response_text_from_responses_api(response: Any) -> str:
    output = getattr(response, "output", None)
    if not output or not output[0].content:
        raise RuntimeError(f"No usable 'output' in OpenAI response: {response}")
    return output[0].content[0].text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_race_strategy(
    course_summary: Dict[str, Any],
    segment_summaries: List[str],
    climb_summaries: List[str],
    climb_block_summaries: List[str],
    athlete_profile: Dict[str, Any],
    *,
    json_only: bool = False,
    model: str = "gpt-4.1-mini",
    max_output_tokens: int = 3500,
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    prompt = build_strategy_prompt(
        course_summary=course_summary,
        segment_summaries=segment_summaries,
        climb_summaries=climb_summaries,
        climb_block_summaries=climb_block_summaries,
        athlete_profile=athlete_profile,
        json_only=json_only,
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are an expert ultra-trail running coach and race strategist."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI BadRequestError: {getattr(e, 'message', str(e))}") from e

    if verbose:
        print("OpenAI response id:", getattr(response, "id", None))

    raw_text = _response_text_from_responses_api(response)
    parsed_json = _extract_json_from_text(raw_text)

    if parsed_json is None:
        raise RuntimeError(
            "LLM did not return parsable JSON. "
            "Try increasing max_output_tokens or inspect raw_text."
        )

    # Coach notes: only for non-json-only runs
    coach_notes = ""
    if not json_only:
        first_brace = raw_text.find("{")
        coach_notes = raw_text[:first_brace].strip() if first_brace > 0 else raw_text.strip()
        if not coach_notes:
            raise RuntimeError("Full strategy generation did not produce coach notes.")

    return (coach_notes if not json_only else raw_text), parsed_json