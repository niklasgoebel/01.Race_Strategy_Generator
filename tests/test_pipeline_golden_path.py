from pathlib import Path

from src.pipeline import PipelineConfig, run_pipeline


def test_pipeline_golden_path_no_llm():
    """
    Golden-path test:
    - real GPX
    - full pipeline
    - LLM skipped
    """

    gpx_path = Path("tests/data/sample_trail.gpx")
    assert gpx_path.exists(), "Sample GPX missing"

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

    # Strategy explicitly skipped
    assert res.strategy_text == ""
    assert res.strategy_data == {}
    assert isinstance(res.strategy_tables, dict)

    # Climb blocks exist
    assert hasattr(res, "climb_blocks")