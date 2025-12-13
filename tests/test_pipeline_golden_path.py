from pathlib import Path

import numpy as np

from src.pipeline import run_pipeline, PipelineConfig


def test_pipeline_golden_path_no_llm():
    """
    Golden-path test:
    - real GPX
    - full pipeline
    - no LLM
    - assert sane outputs
    """

    gpx_path = Path("tests/data/sample_trail.gpx")
    assert gpx_path.exists(), "Sample GPX missing"

    cfg = PipelineConfig(
        gpx_path=str(gpx_path),
        llm_model="SKIP",
        llm_max_output_tokens=0,
    )

    # IMPORTANT: pipeline must not call LLM
    result = run_pipeline(cfg)

    # -----------------------
    # Raw GPX sanity
    # -----------------------
    df = result.df_gpx
    assert not df.empty
    assert "cum_distance" in df.columns
    assert df["cum_distance"].iloc[-1] > 1000  # >1km

    # Elevation sanity
    assert "elev_smooth" in df.columns
    assert df["elev_smooth"].min() > -50  # no insane drops
    assert df["elev_smooth"].max() > 50

    # -----------------------
    # Course summary sanity
    # -----------------------
    summary = result.course_summary
    assert summary["total_distance_km"] > 1
    assert summary["total_gain_m"] > 10
    assert summary["num_segments"] > 1

    # Elevation diagnostics exist
    eq = summary.get("elevation_quality")
    assert isinstance(eq, dict)
    assert "spikes_fixed" in eq

    # -----------------------
    # Segments sanity
    # -----------------------
    seg = result.seg
    assert not seg.empty
    assert seg["distance_km"].sum() > 0

    # No micro-segment explosion
    assert len(seg) < 200

    # -----------------------
    # Key climbs sanity
    # -----------------------
    kc = result.key_climbs
    if not kc.empty:
        assert kc["gain_m"].max() > 20