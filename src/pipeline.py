from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.athlete_profile import get_default_athlete_profile
from src.course_model import build_full_course_model
from src.effort_blocks import summarize_climb_blocks
from src.loaders.gpx_loader import load_gpx_to_df
from src.outputs.output_formatter import (
    make_course_overview_table,
    make_key_climbs_table,
    make_segments_table,
    make_strategy_tables,
)
from src.race_strategy_generator import generate_race_strategy


@dataclass
class PipelineConfig:
    gpx_path: str

    # elevation smoothing controls (optional - auto-determined if not provided)
    savgol_window_length: Optional[int] = None
    savgol_polyorder: Optional[int] = None

    # model controls
    llm_model: str = "gpt-4.1-mini"
    llm_max_output_tokens: int = 2000

    # hard skip (for tests / dev)
    skip_llm: bool = False

    # JSON-only mode is for retries only, not initial generation
    llm_json_only: bool = False

    # helpful during debugging
    llm_verbose: bool = False
    
    # athlete profile (optional, uses default if None)
    athlete_profile: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    # raw + cleaned data
    df_gpx: pd.DataFrame

    # course model
    df_res: pd.DataFrame
    seg: Any
    key_climbs: pd.DataFrame
    climb_blocks: pd.DataFrame

    course_summary: Dict[str, Any]
    segment_summaries: list[str]
    climb_summaries: list[str]
    climb_block_summaries: list[str]

    # strategy
    strategy_text: str
    strategy_data: Dict[str, Any]

    # formatted outputs (UI)
    overview_df: pd.DataFrame
    segments_df: pd.DataFrame
    climbs_df: pd.DataFrame
    strategy_tables: Dict[str, pd.DataFrame]
    
    # time estimates
    segments_with_times: Optional[pd.DataFrame] = None
    finish_time_range: Optional[Dict[str, float]] = None
    aid_station_splits: Optional[pd.DataFrame] = None


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """
    One canonical end-to-end runner.
    This is what the notebook / web app / CLI should call.
    """

    # Use provided profile or fallback to default (needed for athlete-aware segmentation and time estimates)
    athlete_profile = cfg.athlete_profile if cfg.athlete_profile is not None else get_default_athlete_profile()
    
    # 1) Load + clean GPX (robust elevation handling)
    df_gpx = load_gpx_to_df(
        cfg.gpx_path,
        window_length=cfg.savgol_window_length,
        polyorder=cfg.savgol_polyorder,
    )

    # 2) Build full course model with athlete-aware segmentation
    (
        df_res,
        seg,
        key_climbs,
        climb_blocks,
        course_summary,
        segment_summaries,
        climb_summaries,
    ) = build_full_course_model(df_gpx, athlete_profile=athlete_profile)

    # Defensive: ensure we always have a DataFrame
    if climb_blocks is None:
        climb_blocks = pd.DataFrame()

    climb_block_summaries = summarize_climb_blocks(climb_blocks, max_items=10)

    # Attach climb-block summary to course_summary for downstream prompt/UI
    course_summary["climb_blocks_top"] = climb_block_summaries
    course_summary["num_climb_blocks"] = int(len(climb_blocks))

    # Make elevation diagnostics available downstream (from df_gpx.attrs)
    if "elevation_quality" not in course_summary:
        eq = getattr(df_gpx, "attrs", {}).get("elevation_quality")
        if eq is not None:
            course_summary["elevation_quality"] = eq

    # 3) Strategy (LLM or skipped)
    strategy_text = ""
    strategy_data: Dict[str, Any] = {}
    strategy_tables: Dict[str, pd.DataFrame] = {}

    should_skip = bool(cfg.skip_llm) or (str(cfg.llm_model).upper() == "SKIP")

    if not should_skip:
        strategy_text, strategy_data = generate_race_strategy(
            course_summary=course_summary,
            segment_summaries=segment_summaries,
            climb_summaries=climb_summaries,
            climb_block_summaries=climb_block_summaries,
            athlete_profile=athlete_profile,
            segments_df=seg,  # Pass enriched segments dataframe
            climb_blocks_df=climb_blocks,  # Pass climb blocks dataframe
            model=cfg.llm_model,
            max_output_tokens=cfg.llm_max_output_tokens,
            json_only=bool(cfg.llm_json_only),
            verbose=cfg.llm_verbose,
        )
        strategy_tables = make_strategy_tables(strategy_data)

    # 4) Format outputs (tables for UI)
    overview_df = make_course_overview_table(course_summary)
    segments_df = make_segments_table(seg)
    climbs_df = make_key_climbs_table(key_climbs)
    
    # 5) Time estimates (if profile available)
    segments_with_times = None
    finish_time_range = None
    aid_station_splits = None
    
    if athlete_profile:
        from src.time_estimator import (
            estimate_segment_times,
            calculate_finish_time_range,
            format_time_hhmm,
        )
        
        segments_with_times = estimate_segment_times(seg, athlete_profile)
        conservative, expected, aggressive = calculate_finish_time_range(segments_with_times)
        
        finish_time_range = {
            "conservative_min": conservative,
            "conservative_str": format_time_hhmm(conservative),
            "expected_min": expected,
            "expected_str": format_time_hhmm(expected),
            "aggressive_min": aggressive,
            "aggressive_str": format_time_hhmm(aggressive),
        }

    return PipelineResult(
        df_gpx=df_gpx,
        df_res=df_res,
        seg=seg,
        key_climbs=key_climbs,
        climb_blocks=climb_blocks,
        course_summary=course_summary,
        segment_summaries=segment_summaries,
        climb_summaries=climb_summaries,
        climb_block_summaries=climb_block_summaries,
        strategy_text=strategy_text,
        strategy_data=strategy_data,
        overview_df=overview_df,
        segments_df=segments_df,
        climbs_df=climbs_df,
        strategy_tables=strategy_tables,
        segments_with_times=segments_with_times,
        finish_time_range=finish_time_range,
        aid_station_splits=aid_station_splits,
    )