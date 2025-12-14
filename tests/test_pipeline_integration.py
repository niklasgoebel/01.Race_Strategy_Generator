from pathlib import Path

from src.pipeline import PipelineConfig, run_pipeline


def test_run_pipeline_end_to_end_without_llm(tmp_path: Path):
    """
    End-to-end integration test:
    - synthetic GPX
    - elevation spike
    - LLM skipped
    """

    pts = []
    base_lat = 43.0000
    lon = 11.0000

    for i in range(20):
        lat = base_lat + i * 0.001
        ele = 10 + i

        if i == 10:
            ele = 200  # spike

        pts.append(f'<trkpt lat="{lat:.4f}" lon="{lon:.4f}"><ele>{ele}</ele></trkpt>')

    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="pytest" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
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
        skip_llm=True,
    )

    res = run_pipeline(cfg)

    # Core outputs
    assert res.df_gpx is not None and not res.df_gpx.empty
    assert res.df_res is not None and not res.df_res.empty
    assert res.overview_df is not None and not res.overview_df.empty
    assert res.segments_df is not None and not res.segments_df.empty

    # Course summary
    cs = res.course_summary
    assert isinstance(cs, dict)
    assert cs.get("total_distance_km", 0) > 0
    assert "total_gain_m" in cs
    assert "total_loss_m" in cs

    # Elevation diagnostics propagated
    eq = cs.get("elevation_quality")
    assert isinstance(eq, dict)
    assert "missing_frac" in eq

    # Strategy skipped
    assert res.strategy_text == ""
    assert res.strategy_data == {}