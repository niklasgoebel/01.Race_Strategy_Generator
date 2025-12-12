# src/pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.loaders.gpx_loader import load_gpx_to_df
from src.course.course_builder import build_full_course_model
from src.athlete_profile import get_default_athlete_profile
from src.race_strategy_generator import generate_race_strategy
from src.outputs.output_formatter import (
    make_course_overview_table,
    make_segments_table,
    make_key_climbs_table,
    make_strategy_tables,
)


@dataclass
class PipelineConfig:
    gpx_path: str
    elevation_floor_m: float = 200.0
    savgol_window_length: int = 13
    savgol_polyorder: int = 3

    # model controls
    llm_model: str = "gpt-4.1-mini"
    llm_max_output_tokens: int = 2000

    # force JSON-only output (recommended for reliability)
    llm_json_only: bool = True

    # helpful during debugging
    llm_verbose: bool = False


@dataclass
class PipelineResult:
    # raw data
    df_gpx: pd.DataFrame

    # course model
    df_res: pd.DataFrame
    seg: Any
    key_climbs: pd.DataFrame
    course_summary: Dict[str, Any]
    segment_summaries: list[str]
    climb_summaries: list[str]

    # strategy
    strategy_text: str
    strategy_data: Dict[str, Any]

    # formatted outputs (nice for UI)
    overview_df: pd.DataFrame
    segments_df: pd.DataFrame
    climbs_df: pd.DataFrame
    strategy_tables: Dict[str, pd.DataFrame]


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """
    One canonical end-to-end runner.
    This is what the notebook / web app / CLI should call.
    """

    # 1) Load + clean GPX
    df_gpx = load_gpx_to_df(
        cfg.gpx_path,
        elevation_floor_m=cfg.elevation_floor_m,
        window_length=cfg.savgol_window_length,
        polyorder=cfg.savgol_polyorder,
    )

    # 2) Build full course model (segmentation, climbs, summaries)
    df_res, seg, key_climbs, course_summary, segment_summaries, climb_summaries = (
        build_full_course_model(df_gpx)
    )

    # 3) Generate strategy from LLM
    athlete_profile = get_default_athlete_profile()
    strategy_text, strategy_data = generate_race_strategy(
        course_summary=course_summary,
        segment_summaries=segment_summaries,
        climb_summaries=climb_summaries,
        athlete_profile=athlete_profile,
        model=cfg.llm_model,
        max_output_tokens=cfg.llm_max_output_tokens,
        json_only=cfg.llm_json_only,
        verbose=cfg.llm_verbose,
    )
    # generate_race_strategy now raises if JSON can't be parsed, so strategy_data is guaranteed here.

    # 4) Format outputs (tables for UI)
    overview_df = make_course_overview_table(course_summary)

    # IMPORTANT: output_formatter expects the enriched segments DataFrame ("seg"),
    # not the point-by-point df_res.
    segments_df = make_segments_table(seg)
    climbs_df = make_key_climbs_table(key_climbs)

    strategy_tables = make_strategy_tables(strategy_data)

    return PipelineResult(
        df_gpx=df_gpx,
        df_res=df_res,
        seg=seg,
        key_climbs=key_climbs,
        course_summary=course_summary,
        segment_summaries=segment_summaries,
        climb_summaries=climb_summaries,
        strategy_text=strategy_text,
        strategy_data=strategy_data,
        overview_df=overview_df,
        segments_df=segments_df,
        climbs_df=climbs_df,
        strategy_tables=strategy_tables,
    )