# tests/test_smoke.py

import pandas as pd

from src.course_model import build_full_course_model


def test_build_full_course_model_smoke():
    """
    Smoke test:
    - minimal valid input
    - function runs end-to-end
    - outputs have expected basic structure
    """
    df = pd.DataFrame(
        {
            "cum_distance": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "lat": [0, 0, 0, 0, 0, 0],
            "lon": [0, 0, 0, 0, 0, 0],
            "elev_smooth": [0, 1, 2, 3, 4, 5],
        }
    )

    (
        df_res,
        seg,
        key_climbs,
        climb_blocks,
        course_summary,
        segment_summaries,
        climb_summaries,
    ) = build_full_course_model(
        df,
        step_m=20,
        window_m=40,
        min_segment_km=0.02,
    )

    # Basic sanity checks
    assert not df_res.empty
    assert not seg.empty
    assert isinstance(course_summary, dict)
    assert isinstance(segment_summaries, list)
    assert isinstance(climb_summaries, list)

    # New output
    assert climb_blocks is not None


def test_course_summary_distance_monotonic():
    """
    Course summary distance should match last cumulative distance
    """
    df = pd.DataFrame(
        {
            "cum_distance": [0.0, 20.0, 40.0, 60.0],
            "lat": [0, 0, 0, 0],
            "lon": [0, 0, 0, 0],
            "elev_smooth": [0, 0, 0, 0],
        }
    )

    (
        df_res,
        seg,
        _key_climbs,
        _climb_blocks,
        summary,
        _segment_summaries,
        _climb_summaries,
    ) = build_full_course_model(
        df,
        step_m=20,
        window_m=40,
        min_segment_km=0.02,
    )

    expected_km = df["cum_distance"].iloc[-1] / 1000.0
    assert summary["total_distance_km"] == round(expected_km, 1)