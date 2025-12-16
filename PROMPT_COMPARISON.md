# LLM Prompt Comparison: Before vs After

## Before: Text-Based Segments (Hard to Parse)

```
COURSE OVERVIEW (SEGMENTS)
Each row:
  type | start_km–end_km | distance_km | gain/loss | avg_gradient

climb | 0.0–2.3 km | 2.3 km | gain 180 m | loss 5 m | avg gradient 7.8%
runnable | 2.3–4.1 km | 1.8 km | gain 20 m | loss 15 m | avg gradient 0.3%
climb | 4.1–7.5 km | 3.4 km | gain 280 m | loss 12 m | avg gradient 8.2%
descent | 7.5–9.2 km | 1.7 km | gain 8 m | loss 195 m | avg gradient -11.5%
...
```

**Problems:**
- LLM must parse text strings (error-prone)
- No cumulative metrics (can't reason about fatigue)
- No difficulty scores (must calculate mentally)
- No race position context (early/mid/late)

---

## After: Structured JSON (Easy to Reason About)

```json
COURSE SEGMENTS (Structured JSON - 50 key segments)
Use this precise data for analysis. Each segment includes:
- type: climb/descent/runnable
- start_km, end_km, distance_km: location and length
- gain_m, loss_m: elevation change
- avg_gradient: average grade (%)
- cumulative_distance_km: total distance covered so far
- cumulative_gain_m: total elevation gain accumulated (KEY for fatigue modeling)
- difficulty_score: combined metric of gain, gradient, distance (higher = harder)
- race_position: early/mid/late in race (KEY for pacing context)

[
  {
    "type": "climb",
    "start_km": 0.0,
    "end_km": 2.3,
    "distance_km": 2.3,
    "gain_m": 180,
    "loss_m": 5,
    "avg_gradient": 7.8,
    "cumulative_distance_km": 2.3,
    "cumulative_gain_m": 180,
    "difficulty_score": 3.2,
    "race_position": "early"
  },
  {
    "type": "runnable",
    "start_km": 2.3,
    "end_km": 4.1,
    "distance_km": 1.8,
    "gain_m": 20,
    "loss_m": 15,
    "avg_gradient": 0.3,
    "cumulative_distance_km": 4.1,
    "cumulative_gain_m": 200,
    "difficulty_score": 0.4,
    "race_position": "early"
  },
  {
    "type": "climb",
    "start_km": 4.1,
    "end_km": 7.5,
    "distance_km": 3.4,
    "gain_m": 280,
    "loss_m": 12,
    "avg_gradient": 8.2,
    "cumulative_distance_km": 7.5,
    "cumulative_gain_m": 480,
    "difficulty_score": 4.8,
    "race_position": "early"
  },
  {
    "type": "climb",
    "start_km": 42.5,
    "end_km": 46.8,
    "distance_km": 4.3,
    "gain_m": 420,
    "loss_m": 25,
    "avg_gradient": 9.8,
    "cumulative_distance_km": 46.8,
    "cumulative_gain_m": 2850,
    "difficulty_score": 8.2,
    "race_position": "mid"
  }
]
```

**Benefits:**
- ✅ Precise JSON parsing (no string extraction errors)
- ✅ Cumulative gain visible (2850m at 46.8km = high fatigue)
- ✅ Difficulty scores (8.2 vs 3.2 = much harder)
- ✅ Race position (mid = critical phase)

---

## Athlete Capabilities: Before vs After

### Before
```
ATHLETE PROFILE
- VO2max: 57
- Max HR: 202
- Lactate threshold HR: 185
- Goal type: finish strong with smart pacing
```

**LLM must infer:**
- What pace can they sustain?
- When should they power hike vs run?
- How conservative should pacing be?

### After
```
ATHLETE PROFILE
- VO2max: 57
- Max HR: 202
- Lactate threshold HR: 185
- Goal type: finish strong with smart pacing

ATHLETE CAPABILITIES (Derived for Strategy)
- Comfortable flat pace: 4.7 min/km
- Power hike threshold: >8% gradient
- Risk tolerance: low
- Fueling capacity: 70g carbs/hour
- Descent comfort: moderate
```

**Benefits:**
- ✅ Direct actionable metrics
- ✅ No need to infer from raw data
- ✅ Clear pacing thresholds

---

## Coaching Directives: Before vs After

### Before
```
TASK
Generate a race strategy including:
- Pacing guidance using RPE + % of max HR
- Fueling reminders where relevant
```

**Problems:**
- Too open-ended
- No specific requirements
- LLM might miss critical aspects

### After
```
CRITICAL COACHING INSTRUCTIONS:
1. For EACH climb block, specify exact effort level (RPE 1-10 + HR% of max)
2. Identify the 3 most costly sections where athletes typically blow up
3. For each critical section, provide specific pacing cues and bailout strategies
4. Adjust effort recommendations based on cumulative_gain_m:
   - If cumulative_gain_m > 2000m, reduce RPE by 1 point (fatigue accumulation)
   - If cumulative_gain_m > 3000m, reduce RPE by 2 points (high fatigue risk)
5. Power hiking strategy:
   - Recommend power hiking (not running) when gradient > 8%
   - For low risk tolerance: be conservative on steep sections
6. Fueling strategy:
   - Pre-emptive fueling BEFORE major climbs (not during steep climbing)
   - Increase fueling on segments in "late" race_position
7. Use difficulty_score to identify sections requiring extra caution
8. Consider race_position context:
   - "early": Conservative pacing, build rhythm, don't get caught up in excitement
   - "mid": Target effort, maintain discipline, this is where races are won/lost
   - "late": Survival mode, mental toughness, simplified goals
9. For descents: Adjust pacing based on descent_comfort_level (moderate)
10. Include specific km markers for mental checkpoints and motivation
```

**Benefits:**
- ✅ Explicit requirements
- ✅ Fatigue-aware pacing rules
- ✅ Context-specific guidance
- ✅ Athlete-personalized thresholds

---

## Example LLM Reasoning Improvement

### Before (Text Parsing)
**LLM sees:** `climb | 42.5–46.8 km | 4.3 km | gain 420 m | avg gradient 9.8%`

**LLM thinks:** "This is a 4.3km climb with 420m gain at 9.8% grade. It's steep and long. Should probably reduce effort... but how much? And where is this in the race?"

**Result:** Generic advice, might miss fatigue context

### After (JSON + Context)
**LLM sees:**
```json
{
  "type": "climb",
  "start_km": 42.5,
  "end_km": 46.8,
  "distance_km": 4.3,
  "gain_m": 420,
  "cumulative_gain_m": 2850,
  "difficulty_score": 8.2,
  "race_position": "mid",
  "avg_gradient": 9.8
}
```

**Plus directives:** "If cumulative_gain_m > 2000m, reduce RPE by 1"

**LLM thinks:** "This is at 42.5km with 2850m cumulative gain (HIGH fatigue). Difficulty score 8.2 (very hard). Race position 'mid' (critical phase). Gradient 9.8% > 8% threshold = POWER HIKE. Must reduce RPE by 1 due to fatigue."

**Result:** 
- "At km 42.5, you'll have 2850m gain in your legs - expect significant fatigue"
- "This 4.3km climb (difficulty 8.2) is one of the hardest sections"
- "POWER HIKE at 9.8% gradient - do NOT try to run this"
- "Target RPE 4 (reduced from 5 due to cumulative fatigue)"
- "Pre-fuel with gel at km 41 before starting climb"
- "Mental cue: 'This is the race-defining climb - stay disciplined'"

---

## Impact on Strategy Quality

### Precision
- **Before:** "Take it easy on the climbs around km 40"
- **After:** "At km 42.5 (2850m cumulative gain), power hike the 4.3km climb at RPE 4, HR 75-80%"

### Fatigue Awareness
- **Before:** Generic pacing for all climbs
- **After:** Progressive RPE reduction based on cumulative gain

### Athlete Personalization
- **Before:** One-size-fits-all advice
- **After:** "Your 8% power hike threshold means you should hike this 9.8% section"

### Actionability
- **Before:** "Fuel regularly"
- **After:** "Pre-fuel with gel at km 41 before the major climb at km 42.5"

---

## Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| Data Format | Text strings | Structured JSON | 🔥 Easier parsing |
| Fatigue Modeling | None | Cumulative metrics | 🔥 Context-aware pacing |
| Difficulty Assessment | Manual calculation | Pre-computed scores | 🔥 Direct reasoning |
| Athlete Personalization | Inferred from raw data | Pre-computed capabilities | 🔥 Actionable thresholds |
| Coaching Directives | Generic | Explicit & detailed | 🔥 Comprehensive guidance |
| Strategy Precision | Vague km ranges | Exact km markers + metrics | 🔥 Highly specific |

**Result:** LLM can generate significantly more precise, personalized, and actionable race strategies.

