import pandas as pd

from src.course_model import build_full_course_model
from src.race_strategy_generator import _extract_json_from_text


def test_extract_json_from_text_handles_fenced_json():
    txt = """Here you go:
```json
{"a": 1, "b": {"c": 2}}
```"""
    parsed = _extract_json_from_text(txt)
    assert parsed == {"a": 1, "b": {"c": 2}}


def test_build_full_course_model_smoke():
    df = pd.DataFrame(
        {
            "cum_distance": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "lat": [0, 0, 0, 0, 0, 0],
            "lon": [0, 0, 0, 0, 0, 0],
            "elev_smooth": [0, 1, 2, 3, 4, 5],
        }
    )

    df_res, seg, key_climbs, course_summary, segment_summaries, climb_summaries = build_full_course_model(
        df, step_m=20, window_m=40, min_segment_km=0.02
    )

    assert not df_res.empty
    assert "gradient_final" in df_res.columns
    assert not seg.empty
    assert "total_distance_km" in course_summary
    assert isinstance(segment_summaries, list)
    assert isinstance(climb_summaries, list)


def test_course_summary_distance_monotonic():
    df = pd.DataFrame(
        {
            "cum_distance": [0.0, 20.0, 40.0, 60.0],
            "lat": [0, 0, 0, 0],
            "lon": [0, 0, 0, 0],
            "elev_smooth": [0, 0, 0, 0],
        }
    )
    df_res, seg, _, summary, _, _ = build_full_course_model(
        df, step_m=20, window_m=40, min_segment_km=0.02
    )
    assert df_res["cum_distance"].is_monotonic_increasing
    assert summary["total_distance_km"] > 0
