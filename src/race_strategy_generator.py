from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
    segments_df: Optional[pd.DataFrame] = None,
    climb_blocks_df: Optional[pd.DataFrame] = None,
    *,
    json_only: bool = False,
) -> str:
    """
    Build a structured prompt for the LLM.

    If json_only=True, instruct the model to output ONLY valid JSON.
    Otherwise, ask for coach notes (human) then JSON (machine).
    
    Now includes structured JSON data for segments and climb blocks,
    plus athlete capability summary for easier LLM reasoning.
    """
    # Import here to avoid circular dependency
    from src.athlete_profile import calculate_athlete_capabilities
    
    # Calculate athlete capabilities for easier LLM reasoning
    athlete_capabilities = calculate_athlete_capabilities(athlete_profile)
    
    # Prepare segments as JSON (if dataframe provided)
    segments_json = None
    if segments_df is not None and not segments_df.empty:
        # Select key columns for LLM, limit to 50 most important segments
        # Priority: all climbs + evenly spaced others
        df = segments_df.copy()
        
        # Ensure we have the enriched columns
        if "difficulty_score" not in df.columns:
            df["difficulty_score"] = 0.0
        if "cumulative_gain_m" not in df.columns:
            df["cumulative_gain_m"] = df.get("gain_m", 0).cumsum()
        if "race_position" not in df.columns:
            total_km = df["end_km"].max()
            df["race_position"] = df["end_km"].apply(
                lambda x: "early" if x < total_km * 0.3 else "mid" if x < total_km * 0.7 else "late"
            )
        
        # Intelligent sampling: prioritize important segments
        # Priority order:
        # 1. All climbs (most critical for ultra strategy)
        # 2. High-difficulty segments (difficulty_score > threshold)
        # 3. Start/finish segments (always include)
        # 4. Evenly distributed samples from remaining
        
        max_segments = 50
        
        # Priority 1: All climbs
        climbs = df[df["type"] == "climb"].copy()
        
        # Priority 2: High-difficulty non-climb segments
        high_difficulty = df[
            (df["type"] != "climb") & 
            (df["difficulty_score"] > df["difficulty_score"].quantile(0.75))
        ].copy()
        
        # Priority 3: Start and finish segments
        start_finish = df.iloc[[0, -1]].copy() if len(df) > 1 else df.iloc[[0]].copy()
        
        # Combine priorities
        priority_segments = pd.concat([climbs, high_difficulty, start_finish]).drop_duplicates()
        priority_segments = priority_segments.sort_values("start_km")
        
        # Priority 4: Sample remaining segments evenly
        remaining = df[~df.index.isin(priority_segments.index)]
        n_remaining_slots = max(0, max_segments - len(priority_segments))
        
        if n_remaining_slots > 0 and len(remaining) > 0:
            # Even distribution by distance
            remaining = remaining.sort_values("start_km")
            if len(remaining) <= n_remaining_slots:
                sampled_remaining = remaining
            else:
                # Sample evenly by index
                indices = np.linspace(0, len(remaining) - 1, n_remaining_slots, dtype=int)
                sampled_remaining = remaining.iloc[indices]
            
            sampled = pd.concat([priority_segments, sampled_remaining]).sort_values("start_km")
        else:
            # If priorities fill all slots, take top by difficulty
            sampled = priority_segments.nlargest(max_segments, "difficulty_score").sort_values("start_km")
        
        # Select columns for JSON export (include time-of-day if available)
        json_columns = [
            "type", "start_km", "end_km", "distance_km",
            "gain_m", "loss_m", "avg_gradient",
            "cumulative_distance_km", "cumulative_gain_m",
            "difficulty_score", "race_position"
        ]
        
        # Add time-of-day columns if they exist
        if "estimated_time_of_day" in sampled.columns:
            json_columns.extend(["estimated_time_of_day", "time_of_day_category"])
        
        # Convert to JSON-friendly format
        segments_json = sampled[json_columns].round(2).to_dict(orient="records")
    
    # Prepare climb blocks as JSON (if dataframe provided)
    climb_blocks_json = None
    if climb_blocks_df is not None and not climb_blocks_df.empty:
        climb_blocks_json = climb_blocks_df[[
            "block_id", "start_km", "end_km", "distance_km",
            "gain_m", "avg_gradient_pct", "effort_cost",
            "num_runnable_gaps", "longest_runnable_gap_km"
        ]].round(2).to_dict(orient="records")
    
    # Fallback to text summaries if no dataframes
    max_segments = 40
    segment_text = "\n".join(segment_summaries[:max_segments]) if segment_summaries else "-"

    max_climbs = 10
    climb_text = "\n".join(climb_summaries[:max_climbs]) if climb_summaries else "-"

    max_blocks = 10
    climb_blocks_text = "\n".join(climb_block_summaries[:max_blocks]) if climb_block_summaries else "-"

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
      "notes": "string – specific tips for this section",
      "confidence": "string – 'high', 'medium', or 'low'",
      "reasoning": "string – brief explanation of recommendation basis"
    }
  ],

  "pacing_chunks": [
    {
      "start_km": float,
      "end_km": float,
      "terrain_summary": "string",
      "effort_rpe": "string",
      "effort_hr_percent_max": "string",
      "key_focus": "string",
      "confidence": "string – 'high', 'medium', or 'low'"
    }
  ],

  "fueling_plan": {
    "carbs_g_per_hour": int,
    "hydration_notes": "string",
    "confidence": "string – 'high', 'medium', or 'low'",
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

    # Build context with structured data
    athlete_capabilities_text = f"""
- Comfortable flat pace: {athlete_capabilities['flat_comfortable_pace_min_per_km']:.1f} min/km
- Power hike threshold: >{athlete_capabilities['power_hike_threshold_gradient_pct']:.0f}% gradient
- Risk tolerance: {athlete_capabilities['risk_tolerance']}
- Fueling capacity: {athlete_capabilities['fueling_capacity_g_per_hour']}g carbs/hour
- Descent comfort: {athlete_capabilities['descent_comfort_level']}
""".strip()
    
    # Check if time-of-day data is available
    has_time_of_day = segments_json and any("estimated_time_of_day" in seg for seg in segments_json[:1])
    
    # Use JSON segments if available, otherwise fall back to text
    if segments_json:
        time_of_day_desc = """
- estimated_time_of_day: Estimated time when reaching this segment (HH:MM format)
- time_of_day_category: early_morning/late_morning/afternoon/evening/night (KEY for conditions)""" if has_time_of_day else ""
        
        segments_section = f"""
COURSE SEGMENTS (Structured JSON - {len(segments_json)} key segments)
Use this precise data for analysis. Each segment includes:
- type: climb/descent/runnable
- start_km, end_km, distance_km: location and length
- gain_m, loss_m: elevation change
- avg_gradient: average grade (%)
- cumulative_distance_km: total distance covered so far
- cumulative_gain_m: total elevation gain accumulated (KEY for fatigue modeling)
- difficulty_score: combined metric of gain, gradient, distance (higher = harder)
- race_position: early/mid/late in race (KEY for pacing context){time_of_day_desc}

{json.dumps(segments_json, indent=2)}
""".strip()
    else:
        segments_section = f"""
COURSE OVERVIEW (SEGMENTS)
Each row:
  type | start_km–end_km | distance_km | gain/loss | avg_gradient
{segment_text}
""".strip()
    
    # Use JSON climb blocks if available
    if climb_blocks_json:
        climb_blocks_section = f"""
CLIMB BLOCKS (Structured JSON - Sustained Effort Analysis)
These represent sustained climbing efforts that may include SHORT runnable interruptions.
IMPORTANT: Treat these as continuous "costly" effort with LIMITED recovery.

Key metrics:
- effort_cost: Combined difficulty score (gain × gradient × distance factors)
- num_runnable_gaps: Short flat sections within the block (do NOT provide full recovery)
- longest_runnable_gap_km: Maximum flat interruption length

{json.dumps(climb_blocks_json, indent=2)}
""".strip()
    else:
        climb_blocks_section = f"""
CLIMB BLOCKS (SUSTAINED EFFORT STRUCTURE)
These represent sustained climbing efforts that may include SHORT runnable interruptions.
IMPORTANT: Treat these as continuous "costly" effort with LIMITED recovery.

{climb_blocks_text}
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
- Preferred fuel type: {athlete_profile.get("fuel_type")}

ATHLETE CAPABILITIES (Derived for Strategy - USE THESE for recommendations)
{athlete_capabilities_text}

NOTE: The fueling capacity shown above is an ESTIMATE of gut training capacity. You should determine the optimal carbs/hour based on:
1. Race duration and intensity
2. Environmental conditions (heat increases needs)
3. Individual tolerance (use capacity as upper bound)
4. Timing relative to effort (fuel before climbs, not during hard efforts)

{segments_section}

RACE STRUCTURE (High-Level Overview)
{chr(10).join(course_summary.get("macro_structure", []) or ["-"])}

KEY CLIMBS (MOST IMPORTANT FEATURES)
{climb_text}

{climb_blocks_section}
""".strip()

    # Build coaching directives based on athlete capabilities
    hike_threshold = athlete_capabilities['power_hike_threshold_gradient_pct']
    risk_level = athlete_capabilities['risk_tolerance']
    
    # Add time-of-day specific directives if data available
    time_of_day_directives = ""
    if has_time_of_day:
        time_of_day_directives = """
11. Time-of-day considerations (if time_of_day_category present):
   - "early_morning": Cool temps, fresh legs, good conditions for hard efforts
   - "late_morning": Warming up, start increasing hydration
   - "afternoon": HOT - increase hydration by 20-30%, seek shade at aid stations, reduce effort on exposed sections
   - "evening": Cooling down, energy may dip, focus on fueling and mental focus
   - "night": Cold, headlamp required, slower pace natural, extra layers needed
12. Flag any major climbs hitting during "afternoon" for heat/hydration warnings
13. Adjust fueling based on time-of-day (more frequent in heat, energy dips in evening)"""
    
    coaching_directives = f"""
CRITICAL COACHING INSTRUCTIONS:
1. For EACH climb block, specify exact effort level (RPE 1-10 + HR% of max)
2. Identify the 3 most costly sections where athletes typically blow up
3. For each critical section, provide specific pacing cues and bailout strategies
4. Adjust effort recommendations based on cumulative_gain_m:
   - If cumulative_gain_m > 2000m, reduce RPE by 1 point (fatigue accumulation)
   - If cumulative_gain_m > 3000m, reduce RPE by 2 points (high fatigue risk)
5. Power hiking strategy:
   - Recommend power hiking (not running) when gradient > {hike_threshold:.0f}%
   - For {risk_level} risk tolerance: {'be conservative on steep sections' if risk_level == 'low' else 'push harder on climbs but watch for blow-up' if risk_level == 'high' else 'balance effort across the race'}
6. Fueling strategy:
   - Pre-emptive fueling BEFORE major climbs (not during steep climbing)
   - Increase fueling on segments in "late" race_position
   - Flag any segments where cumulative effort requires extra nutrition
7. Use difficulty_score to identify sections requiring extra caution
8. Consider race_position context:
   - "early": Conservative pacing, build rhythm, don't get caught up in start excitement
   - "mid": Target effort, maintain discipline, this is where races are won/lost
   - "late": Survival mode, mental toughness, simplified goals
9. For descents: Adjust pacing based on descent_comfort_level ({athlete_capabilities['descent_comfort_level']})
10. Include specific km markers for mental checkpoints and motivation{time_of_day_directives}

CONFIDENCE & REASONING REQUIREMENTS:
- For EACH recommendation in critical_sections and pacing_chunks, include:
  * confidence: "high" (well-established pacing principles, clear data), 
               "medium" (reasonable inference, some uncertainty), 
               "low" (speculative, limited data, or unusual conditions)
  * reasoning: Brief explanation of WHY this recommendation is made (1-2 sentences)
  
- High confidence examples:
  * "Power hike gradient >10%" (established ultra running principle)
  * "Reduce RPE by 1 at 2000m cumulative gain" (based on fatigue modeling)
  * "Increase hydration in afternoon heat" (well-known physiological need)
  
- Medium confidence examples:
  * "Specific HR% target for unknown athlete" (estimated from VO2max)
  * "Exact fueling timing" (individual variation high)
  
- Low confidence examples:
  * "Technical terrain speed without course details" (missing information)
  * "Weather-dependent strategies without forecast" (speculation)
  
- Be honest about uncertainty - athletes need to know what's reliable vs. speculative
""".strip()

    if json_only:
        task_block = f"""
TASK
You MUST output ONLY a valid JSON object following the schema below.
- No markdown fences. No extra commentary. Only JSON.
- Use double quotes for all strings.
- No trailing commas.

{coaching_directives}

Return EXACTLY this schema (same keys, same nesting, same field types):
{json_schema}
""".strip()
    else:
        task_block = f"""
TASK
You MUST output in TWO PARTS, in this exact order:

{coaching_directives}

PART 1 — COACH NOTES (human-readable)
- Write 8–15 bullet points.
- Short, concrete, actionable.
- Reference specific course sections (km ranges, climbs, climb blocks).
- Include pacing guidance using RPE + % of max HR.
- Include fueling reminders where relevant.
- Highlight the 3 most critical/dangerous sections.
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
    segments_df: Optional[pd.DataFrame] = None,
    climb_blocks_df: Optional[pd.DataFrame] = None,
    *,
    json_only: bool = False,
    model: str = "gpt-4.1-mini",
    max_output_tokens: int = 3500,
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate race strategy using LLM.
    
    Now accepts optional segments_df and climb_blocks_df for richer
    structured data in the prompt.
    """
    prompt = build_strategy_prompt(
        course_summary=course_summary,
        segment_summaries=segment_summaries,
        climb_summaries=climb_summaries,
        climb_block_summaries=climb_block_summaries,
        athlete_profile=athlete_profile,
        segments_df=segments_df,
        climb_blocks_df=climb_blocks_df,
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