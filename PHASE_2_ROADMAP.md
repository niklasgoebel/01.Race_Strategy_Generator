# Phase 2 & 3 Roadmap - Future Improvements

## Overview
Phase 1 (Quick Wins) is complete. This document outlines the recommended next steps for Phase 2 (Segmentation Improvements) and Phase 3 (Advanced Features).

---

## Phase 2: Segmentation Improvements (4-5 hours)

### 2.1 Athlete-Aware Gradient Thresholds ⭐ HIGH PRIORITY

**Current Issue:**
- Fixed ±3% thresholds for climb/descent/runnable classification
- Same segmentation for elite vs beginner athletes
- Doesn't account for athlete capabilities

**Proposed Solution:**
```python
# src/course_model.py

def classify_gradient_athlete_aware(
    g: float, 
    athlete_capabilities: Dict[str, Any]
) -> str:
    """
    Classify gradient based on athlete-specific thresholds.
    """
    hike_threshold = athlete_capabilities.get('power_hike_threshold_gradient_pct', 8.0)
    
    # Climb threshold = slightly below hike threshold
    # (can still run, but it's a climb)
    climb_threshold = hike_threshold * 0.6  # e.g., 4.8% for 8% hiker
    
    # Descent threshold (less athlete-dependent)
    descent_threshold = -4.0
    
    if g > climb_threshold:
        return "climb"
    elif g < descent_threshold:
        return "descent"
    else:
        return "runnable"
```

**Implementation Steps:**
1. Modify `build_full_course_model()` to accept optional `athlete_profile`
2. Calculate capabilities at start of segmentation
3. Use athlete-aware thresholds in `classify_gradient()`
4. Update tests to verify different segmentation for different athletes

**Expected Impact:**
- Elite athletes: More "runnable" sections (can run steeper grades)
- Beginner athletes: More "climb" sections (need to hike earlier)
- Better match between segmentation and actual athlete behavior

---

### 2.2 Multi-Scale Gradient Analysis ⭐ MEDIUM PRIORITY

**Current Issue:**
- Single 100m gradient window
- Short steep pitches get averaged out
- Long gradual climbs lack context

**Proposed Solution:**
```python
# src/course_model.py

def compute_multiscale_gradients(
    df_res: pd.DataFrame,
    windows: List[int] = [50, 100, 500, 1000]
) -> pd.DataFrame:
    """
    Compute gradients at multiple scales for richer analysis.
    """
    for window_m in windows:
        col_name = f"gradient_{window_m}m"
        df_res[col_name] = compute_gradient_at_window(df_res, window_m)
    
    # Also compute gradient variability (how much it changes)
    df_res["gradient_variability"] = (
        df_res["gradient_50m"].rolling(20).std()
    )
    
    return df_res
```

**Usage in Prompt:**
```python
# For each segment, include multi-scale context
segment_enriched = {
    "start_km": 15.3,
    "avg_gradient": 8.2,
    "gradient_50m_max": 15.3,  # Steepest 50m section
    "gradient_500m_avg": 7.8,  # Overall 500m trend
    "gradient_variability": 2.1,  # How much it varies (high = inconsistent)
}
```

**LLM Prompt Enhancement:**
```
"This 3.4km climb averages 8.2%, but includes 50m sections up to 15% 
(high variability = inconsistent gradient). Prepare for steep surges."
```

**Implementation Steps:**
1. Add `compute_multiscale_gradients()` function
2. Store multi-scale data in segments dataframe
3. Include in JSON sent to LLM
4. Update prompt to explain how to use multi-scale data

**Expected Impact:**
- More nuanced pacing advice
- Better identification of "hidden" steep sections
- Improved mental preparation for variable terrain

---

### 2.3 Hierarchical Segmentation ⭐ MEDIUM PRIORITY

**Current Issue:**
- Merging segments <400m loses micro-terrain details
- LLM gets either too much detail or too little

**Proposed Solution:**
```python
# src/course_model.py

def build_hierarchical_segments(seg: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create both micro (detailed) and macro (high-level) segments.
    """
    # Micro segments: Keep all details (no merging)
    micro_segments = seg.copy()
    
    # Macro segments: Merge aggressively for high-level view
    macro_segments = _merge_short_segments(seg, min_segment_km=1.0)
    
    return {
        "micro": micro_segments,
        "macro": macro_segments,
    }
```

**Usage in Prompt:**
```
MACRO SEGMENTS (High-level race structure):
- 0-12km: Rolling terrain with 3 moderate climbs
- 12-28km: Major climb block (1200m gain)
- 28-45km: Technical descent + valley runnable
- 45-60km: Final climb sequence (800m gain)

MICRO SEGMENTS (Detailed analysis):
[JSON with all 150+ segments for precise reasoning]
```

**Implementation Steps:**
1. Create hierarchical segmentation function
2. Pass both levels to LLM
3. Instruct LLM to use macro for overview, micro for details
4. Update prompt structure

**Expected Impact:**
- LLM gets both "big picture" and detailed view
- Better strategic planning (macro) + tactical execution (micro)
- Reduced risk of losing important micro-terrain features

---

### 2.4 Segment Difficulty Calibration ⭐ LOW PRIORITY

**Current Issue:**
- Difficulty score formula is reasonable but not calibrated to real-world data

**Proposed Solution:**
- Collect feedback on actual perceived difficulty
- Adjust formula weights based on athlete reports
- Add terrain type factor (technical vs non-technical)

```python
def calculate_difficulty_score_v2(
    gain_m: float,
    gradient_pct: float,
    distance_km: float,
    terrain_type: str = "trail"  # trail/road/technical
) -> float:
    """
    Calibrated difficulty score based on athlete feedback.
    """
    base = (gain_m / 100)
    gradient_factor = 1 + (gradient_pct / 10)
    distance_factor = 1 + (distance_km / 5)
    
    # Terrain multiplier
    terrain_multipliers = {
        "road": 1.0,
        "trail": 1.15,
        "technical": 1.35,
    }
    terrain_factor = terrain_multipliers.get(terrain_type, 1.0)
    
    return base * gradient_factor * distance_factor * terrain_factor
```

---

## Phase 3: Advanced Features (1-2 days)

### 3.1 Adaptive Smoothing ⭐ MEDIUM PRIORITY

**Current Issue:**
- Fixed Savitzky-Golay parameters (window=13, poly=3)
- Not adaptive to GPX resolution or terrain type

**Proposed Solution:**
```python
# src/elevation.py

def adaptive_smoothing_params(
    df: pd.DataFrame,
    total_distance_km: float
) -> Dict[str, int]:
    """
    Determine optimal smoothing parameters based on data characteristics.
    """
    points_per_km = len(df) / total_distance_km
    
    # High resolution (>200 pts/km) = more smoothing needed
    if points_per_km > 200:
        window = 21
        polyorder = 3
    elif points_per_km > 100:
        window = 13
        polyorder = 3
    else:
        # Low resolution = less smoothing (preserve detail)
        window = 7
        polyorder = 2
    
    return {"window_length": window, "polyorder": polyorder}
```

**Two-Pass Terrain-Aware Smoothing:**
```python
def terrain_aware_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply adaptive smoothing: less on steep sections, more on flats.
    """
    # Pass 1: Light smooth to identify terrain zones
    df["elev_light"] = savgol_filter(df["elevation"], 7, 2)
    df["gradient_rough"] = compute_gradient(df["elev_light"])
    
    # Pass 2: Adaptive smoothing based on gradient
    smoothed = []
    for i, row in df.iterrows():
        if abs(row["gradient_rough"]) > 8:
            # Steep section: less smoothing (preserve sharp features)
            window = 7
        else:
            # Flat section: more smoothing (reduce noise)
            window = 15
        
        # Apply local smoothing
        start = max(0, i - window // 2)
        end = min(len(df), i + window // 2)
        local_smooth = savgol_filter(df["elevation"].iloc[start:end], window, 2)
        smoothed.append(local_smooth[window // 2])
    
    df["elev_smooth"] = smoothed
    return df
```

---

### 3.2 Time-of-Day Integration ⭐ HIGH PRIORITY

**Current Issue:**
- No consideration of when segments will be reached
- Can't adjust for heat, sun exposure, darkness

**Proposed Solution:**
```python
# src/course_model.py

def add_time_of_day_estimates(
    seg: pd.DataFrame,
    athlete_profile: Dict[str, Any],
    race_start_time: str = "07:00"  # e.g., "07:00" for 7am start
) -> pd.DataFrame:
    """
    Estimate time-of-day for each segment based on pacing model.
    """
    from datetime import datetime, timedelta
    from src.time_estimator import estimate_segment_times
    
    # Get time estimates
    seg_with_times = estimate_segment_times(seg, athlete_profile)
    
    # Convert to time-of-day
    start_dt = datetime.strptime(race_start_time, "%H:%M")
    
    seg["estimated_time_of_day"] = seg_with_times["cumulative_time_min"].apply(
        lambda mins: (start_dt + timedelta(minutes=mins)).strftime("%H:%M")
    )
    
    seg["time_of_day_category"] = seg["estimated_time_of_day"].apply(
        lambda t: categorize_time_of_day(t)
    )
    
    return seg

def categorize_time_of_day(time_str: str) -> str:
    """Categorize time of day for strategy purposes."""
    hour = int(time_str.split(":")[0])
    
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
```

**Usage in Prompt:**
```json
{
  "type": "climb",
  "start_km": 42.5,
  "estimated_time_of_day": "14:30",
  "time_of_day_category": "afternoon",
  "notes": "Hot sun exposure - increase hydration"
}
```

**LLM Directive:**
```
When time_of_day_category is "afternoon":
- Increase hydration recommendations by 20%
- Warn about heat exposure on exposed climbs
- Suggest seeking shade at aid stations
```

---

### 3.3 Confidence Scores in LLM Output ⭐ LOW PRIORITY

**Current Issue:**
- LLM outputs all recommendations as facts
- No indication of certainty level

**Proposed Solution:**
Update JSON schema:
```json
{
  "critical_sections": [
    {
      "label": "Climb #1",
      "confidence": "high",
      "reasoning": "Based on typical ultra pacing models and 8% gradient",
      "effort_rpe": "4-5",
      "notes": "Power hike steep sections over 10%"
    }
  ]
}
```

**LLM Directive:**
```
For each recommendation, include:
- confidence: "high" (well-established pacing principles), 
             "medium" (reasonable inference), 
             "low" (speculative based on limited data)
- reasoning: Brief explanation of why this recommendation is made
```

---

### 3.4 Elevation Gain/Loss Validation ⭐ LOW PRIORITY

**Current Issue:**
- No sanity check on total elevation gain/loss
- GPS drift can create unrealistic totals

**Proposed Solution:**
```python
# src/elevation.py

def validate_elevation_metrics(
    total_gain_m: float,
    total_loss_m: float,
    distance_km: float,
    course_type: str = "trail"
) -> Dict[str, Any]:
    """
    Validate elevation metrics and flag anomalies.
    """
    gain_per_km = total_gain_m / distance_km
    loss_per_km = total_loss_m / distance_km
    
    # Expected ranges by course type
    ranges = {
        "road": (0, 30),      # 0-30m gain per km
        "trail": (20, 80),    # 20-80m gain per km
        "mountain": (50, 150) # 50-150m gain per km
    }
    
    min_expected, max_expected = ranges.get(course_type, (0, 150))
    
    warnings = []
    if gain_per_km > max_expected:
        warnings.append(
            f"Elevation gain ({gain_per_km:.0f}m/km) is unusually high. "
            f"GPS drift or data quality issues likely."
        )
    
    if gain_per_km < min_expected and course_type != "road":
        warnings.append(
            f"Elevation gain ({gain_per_km:.0f}m/km) is unusually low for {course_type}. "
            f"Check GPX data quality."
        )
    
    # Check gain/loss balance (should be similar for loop courses)
    imbalance = abs(total_gain_m - total_loss_m)
    if imbalance > 200:
        warnings.append(
            f"Gain/loss imbalance: {imbalance:.0f}m difference. "
            f"Point-to-point course or data quality issue."
        )
    
    return {
        "gain_per_km": gain_per_km,
        "loss_per_km": loss_per_km,
        "warnings": warnings,
        "quality": "good" if not warnings else "check_needed"
    }
```

---

## Implementation Priority Matrix

| Feature | Impact | Effort | Priority | Timeline |
|---------|--------|--------|----------|----------|
| Time-of-day integration | 🔥 HIGH | Medium | ⭐ 1 | Week 1 |
| Athlete-aware thresholds | 🔥 HIGH | Low | ⭐ 2 | Week 1 |
| Multi-scale gradients | 🔴 MEDIUM | Medium | ⭐ 3 | Week 2 |
| Hierarchical segmentation | 🔴 MEDIUM | Medium | ⭐ 4 | Week 2 |
| Adaptive smoothing | 🔴 MEDIUM | Medium | ⭐ 5 | Week 3 |
| Confidence scores | 🟡 LOW | Low | ⭐ 6 | Week 3 |
| Elevation validation | 🟡 LOW | Low | ⭐ 7 | Week 3 |
| Difficulty calibration | 🟡 LOW | High | ⭐ 8 | Future |

---

## Testing Strategy for Phase 2/3

### Unit Tests
```python
# tests/test_athlete_aware_segmentation.py
def test_elite_vs_beginner_segmentation():
    """Elite athletes should have more runnable segments."""
    elite_profile = {"vo2max": 65, "experience": "elite"}
    beginner_profile = {"vo2max": 45, "experience": "beginner"}
    
    elite_seg = build_course_model(gpx, elite_profile)
    beginner_seg = build_course_model(gpx, beginner_profile)
    
    elite_runnable_pct = (elite_seg["type"] == "runnable").mean()
    beginner_runnable_pct = (beginner_seg["type"] == "runnable").mean()
    
    assert elite_runnable_pct > beginner_runnable_pct
```

### Integration Tests
```python
# tests/test_multiscale_gradients.py
def test_multiscale_gradient_detection():
    """Should detect short steep pitches within longer climbs."""
    df_res = compute_multiscale_gradients(df)
    
    # Find segment with high 50m gradient but moderate 500m gradient
    steep_pitch = df_res[
        (df_res["gradient_50m"] > 12) & 
        (df_res["gradient_500m"] < 6)
    ]
    
    assert len(steep_pitch) > 0, "Should detect steep pitches"
```

### Validation Tests
```python
# tests/test_elevation_validation.py
def test_elevation_validation_flags_anomalies():
    """Should flag unrealistic elevation gain."""
    validation = validate_elevation_metrics(
        total_gain_m=10000,  # Unrealistic for 50km
        distance_km=50,
        course_type="trail"
    )
    
    assert validation["quality"] == "check_needed"
    assert len(validation["warnings"]) > 0
```

---

## Success Metrics

### Phase 2
- [ ] Athlete-aware segmentation produces different results for different profiles
- [ ] Multi-scale gradients detect steep pitches missed by single-scale
- [ ] Hierarchical segmentation provides both overview and detail
- [ ] All existing tests still pass

### Phase 3
- [ ] Time-of-day estimates included in all strategies
- [ ] LLM adjusts recommendations based on time-of-day
- [ ] Adaptive smoothing improves elevation quality scores
- [ ] Confidence scores help users understand recommendation certainty

---

## Conclusion

Phase 1 laid the foundation with structured data and enriched metrics. Phase 2/3 will build on this to create even more intelligent, context-aware, and athlete-specific strategies.

**Recommended approach:** Implement features incrementally, validate with real GPX files, and gather user feedback before moving to next feature.

