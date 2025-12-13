# tests/test_pipeline_integration.py

from pathlib import Path

from src.pipeline import PipelineConfig, run_pipeline


def test_run_pipeline_end_to_end_without_llm(tmp_path: Path):
    # Build a small-but-realistic GPX (~2 km), including one elevation spike.
    # Lat step of 0.001 ≈ 111m. With 20 points → ~2.1 km total distance.
    pts = []
    base_lat = 43.0000
    lon = 11.0000

    for i in range(20):
        lat = base_lat + i * 0.001
        ele = 10 + i  # gentle rise

        # Insert one obvious spike to ensure elevation cleaner is exercised
        if i == 10:
            ele = 200

        pts.append(f'<trkpt lat="{lat:.4f}" lon="{lon:.4f}"><ele>{ele}</ele></trkpt>')

    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="pytest" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Test Track</name>
    <trkseg>
      {"".join(pts)}
    </trkseg>
  </trk>
</gpx>
"""

    gpx_path = tmp_path / "test.gpx"
    gpx_path.write_text(gpx_content, encoding="utf-8")

    cfg = PipelineConfig(
        gpx_path=str(gpx_path),
        savgol_window_length=13,
        savgol_polyorder=3,
        skip_llm=True,  # key: no OpenAI call
    )

    res = run_pipeline(cfg)

    # Core outputs exist
    assert res.df_gpx is not None and not res.df_gpx.empty
    assert res.df_res is not None and not res.df_res.empty
    assert res.overview_df is not None and not res.overview_df.empty

    # Segments should exist for a ~2km course
    assert res.segments_df is not None and not res.segments_df.empty

    # Course summary is populated
    assert isinstance(res.course_summary, dict)
    assert res.course_summary.get("total_distance_km", 0) > 0
    assert "total_gain_m" in res.course_summary
    assert "total_loss_m" in res.course_summary

    # Elevation quality should be present (from Session 8)
    eq = res.course_summary.get("elevation_quality")
    assert isinstance(eq, dict)
    assert "missing_frac" in eq
    assert "spikes_fixed" in eq

    # Strategy is skipped
    assert res.strategy_text == ""
    assert res.strategy_data == {}
    assert res.strategy_tables == {}