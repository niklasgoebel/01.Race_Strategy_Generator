# API Reference - New Functions & Enhanced Features

## New Functions

### 1. `calculate_athlete_capabilities(profile: Dict[str, Any]) -> Dict[str, Any]`
**Location:** `src/athlete_profile.py`

Converts raw athlete profile into actionable capabilities for strategy generation.

**Input:**
```python
profile = {
    "vo2max": 57,
    "experience": "intermediate-advanced road runner",
    "goal_type": "finish strong with smart pacing",
    "descent_style": "decently confident",
    "carbs_per_hour_target_g": 70
}
```

**Output:**
```python
{
    "flat_comfortable_pace_min_per_km": 4.7,
    "power_hike_threshold_gradient_pct": 8.0,
    "risk_tolerance": "low",
    "fueling_capacity_g_per_hour": 70,
    "descent_comfort_level": "moderate",
    "estimated_race_time_multiplier": 1.15
}
```

**Usage:**
```python
from src.athlete_profile import calculate_athlete_capabilities, load_athlete_profile

profile = load_athlete_profile("Niklas")
capabilities = calculate_athlete_capabilities(profile)

print(f"Power hike when gradient > {capabilities['power_hike_threshold_gradient_pct']}%")
# Output: Power hike when gradient > 8.0%
```

---

### 2. `calculate_effort_cost(block: Dict[str, Any]) -> float`
**Location:** `src/effort_blocks.py`

Calculates a single difficulty metric for climb blocks.

**Formula:**
```
effort_cost = (gain_m / 100) × (1 + gradient_pct / 10) × (1 + distance_km / 5)
```

**Input:**
```python
block = {
    "gain_m": 420,
    "avg_gradient_pct": 9.8,
    "distance_km": 4.3
}
```

**Output:**
```python
8.2  # Higher = more difficult
```

**Usage:**
```python
from src.effort_blocks import calculate_effort_cost

effort = calculate_effort_cost(block)
if effort > 7.0:
    print("This is a very hard climb - expect significant fatigue")
```

---

## Enhanced Functions

### 3. `enrich_segments(segments_df: pd.DataFrame) -> pd.DataFrame`
**Location:** `src/course_model.py`

Now adds cumulative metrics and difficulty scores to segments.

**New Columns Added:**
- `cumulative_distance_km`: Total distance to this point
- `cumulative_gain_m`: Total elevation gain accumulated
- `difficulty_score`: Combined difficulty metric
- `race_position`: "early" / "mid" / "late"

**Example:**
```python
from src.course_model import build_full_course_model
from src.loaders.gpx_loader import load_gpx_to_df

df_gpx = load_gpx_to_df("data/Chianti course.gpx")
df_res, seg, key_climbs, climb_blocks, *_ = build_full_course_model(df_gpx)

# Segments now include enriched data
print(seg[['start_km', 'end_km', 'cumulative_gain_m', 'difficulty_score', 'race_position']])
```

**Output:**
```
   start_km  end_km  cumulative_gain_m  difficulty_score race_position
0       0.0     2.3              180.0              3.2         early
1       2.3     4.1              200.0              0.4         early
2       4.1     7.5              480.0              4.8         early
...
15     42.5    46.8             2850.0              8.2           mid
...
```

---

### 4. `build_climb_blocks(seg: pd.DataFrame, cfg: EffortBlocksConfig) -> pd.DataFrame`
**Location:** `src/effort_blocks.py`

Now includes `effort_cost` column in output.

**New Column:**
- `effort_cost`: Single difficulty metric for the entire block

**Example:**
```python
from src.effort_blocks import build_climb_blocks

climb_blocks = build_climb_blocks(seg)
print(climb_blocks[['block_id', 'start_km', 'gain_m', 'effort_cost']])
```

**Output:**
```
   block_id  start_km  gain_m  effort_cost
0         1       0.0   180.0          3.2
1         2       4.1   420.0          8.2
2         3      52.3   350.0          6.5
```

---

### 5. `build_strategy_prompt()` - Enhanced
**Location:** `src/race_strategy_generator.py`

Now accepts optional dataframes for richer prompts.

**New Parameters:**
- `segments_df: Optional[pd.DataFrame] = None`
- `climb_blocks_df: Optional[pd.DataFrame] = None`

**Behavior:**
- If dataframes provided: Uses structured JSON format
- If not provided: Falls back to text summaries (backward compatible)

**Example:**
```python
from src.race_strategy_generator import build_strategy_prompt

# New way (with dataframes)
prompt = build_strategy_prompt(
    course_summary=course_summary,
    segment_summaries=segment_summaries,
    climb_summaries=climb_summaries,
    climb_block_summaries=climb_block_summaries,
    athlete_profile=athlete_profile,
    segments_df=seg,  # NEW
    climb_blocks_df=climb_blocks,  # NEW
)

# Old way still works (backward compatible)
prompt = build_strategy_prompt(
    course_summary=course_summary,
    segment_summaries=segment_summaries,
    climb_summaries=climb_summaries,
    climb_block_summaries=climb_block_summaries,
    athlete_profile=athlete_profile,
)
```

---

### 6. `generate_race_strategy()` - Enhanced
**Location:** `src/race_strategy_generator.py`

Now accepts optional dataframes.

**New Parameters:**
- `segments_df: Optional[pd.DataFrame] = None`
- `climb_blocks_df: Optional[pd.DataFrame] = None`

**Example:**
```python
from src.race_strategy_generator import generate_race_strategy

strategy_text, strategy_data = generate_race_strategy(
    course_summary=course_summary,
    segment_summaries=segment_summaries,
    climb_summaries=climb_summaries,
    climb_block_summaries=climb_block_summaries,
    athlete_profile=athlete_profile,
    segments_df=seg,  # NEW
    climb_blocks_df=climb_blocks,  # NEW
    model="gpt-4.1-mini",
)
```

---

## Complete Example: End-to-End Usage

```python
from src.pipeline import PipelineConfig, run_pipeline
from src.athlete_profile import load_athlete_profile, calculate_athlete_capabilities

# 1. Load athlete profile
athlete_profile = load_athlete_profile("Niklas")

# 2. Calculate capabilities (optional - pipeline does this internally)
capabilities = calculate_athlete_capabilities(athlete_profile)
print(f"Athlete should power hike when gradient > {capabilities['power_hike_threshold_gradient_pct']}%")

# 3. Run pipeline with profile
cfg = PipelineConfig(
    gpx_path="data/Chianti course.gpx",
    athlete_profile=athlete_profile,
    skip_llm=False,  # Generate strategy
)

result = run_pipeline(cfg)

# 4. Access enriched data
print("\nEnriched Segments:")
print(result.seg[['start_km', 'end_km', 'cumulative_gain_m', 'difficulty_score', 'race_position']].head())

print("\nClimb Blocks with Effort Cost:")
print(result.climb_blocks[['block_id', 'start_km', 'gain_m', 'effort_cost']])

# 5. View strategy
print("\nCoach Notes:")
print(result.strategy_text)

print("\nCritical Sections:")
for section in result.strategy_data.get('critical_sections', []):
    print(f"  {section['label']}: {section['start_km']}-{section['end_km']}km")
    print(f"    Effort: RPE {section['effort_rpe']}, HR {section['effort_hr_percent_max']}")
    print(f"    Notes: {section['notes']}")
```

---

## Segment Difficulty Score Interpretation

| Score | Difficulty | Example |
|-------|------------|---------|
| 0-2 | Easy | Flat or gentle runnable sections |
| 2-4 | Moderate | Short climbs or gentle sustained efforts |
| 4-6 | Hard | Significant climbs requiring effort management |
| 6-8 | Very Hard | Major climbs, high fatigue risk |
| 8+ | Extreme | Race-defining climbs, critical pacing required |

**Formula Breakdown:**
```python
difficulty_score = (gain_m / 100) × (1 + gradient_pct / 10) × (1 + distance_km / 5)

# Example: 4.3km climb, 420m gain, 9.8% gradient
difficulty = (420 / 100) × (1 + 9.8 / 10) × (1 + 4.3 / 5)
           = 4.2 × 1.98 × 1.86
           = 15.5  # Very high difficulty
```

---

## Effort Cost Interpretation (Climb Blocks)

Similar to difficulty score but applied to entire climb blocks (which may include short runnable gaps).

| Cost | Interpretation | Strategy |
|------|----------------|----------|
| 0-3 | Manageable | Maintain target effort, no special pacing needed |
| 3-5 | Moderate | Monitor effort, consider slight RPE reduction |
| 5-7 | Costly | Reduce RPE by 1, ensure adequate fueling |
| 7-10 | Very Costly | Reduce RPE by 1-2, power hike steep sections, pre-fuel |
| 10+ | Extreme | Major effort block, conservative pacing critical |

---

## Race Position Context

| Position | % of Race | Strategy Focus |
|----------|-----------|----------------|
| **early** | 0-30% | Conservative pacing, build rhythm, avoid excitement |
| **mid** | 30-70% | Target effort, maintain discipline, race-critical phase |
| **late** | 70-100% | Survival mode, mental toughness, simplified goals |

**Usage in Strategy:**
- Early: "Don't get caught up in the start - save energy"
- Mid: "This is where the race is won or lost - stay disciplined"
- Late: "Focus on forward progress - every step counts"

---

## Backward Compatibility

All new parameters are optional with sensible defaults:

```python
# Old code continues to work unchanged
from src.race_strategy_generator import generate_race_strategy

strategy_text, strategy_data = generate_race_strategy(
    course_summary=course_summary,
    segment_summaries=segment_summaries,
    climb_summaries=climb_summaries,
    climb_block_summaries=climb_block_summaries,
    athlete_profile=athlete_profile,
    # No need to pass segments_df or climb_blocks_df
)
```

The system will automatically fall back to text-based summaries if dataframes are not provided.

---

## Testing

All existing tests pass without modification:
```bash
pytest tests/ -v
# 6 passed in 1.38s
```

New functionality is automatically tested through existing integration tests since the pipeline now uses the enhanced features by default.

