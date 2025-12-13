from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.loaders.gpx_loader import load_gpx_to_df
from src.course_model import build_full_course_model
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

    # elevation smoothing controls (advanced)
    savgol_window_length: int = 13
    savgol_polyorder: int = 3

    # model controls
    llm_model: str = "gpt-4.1-mini"
    llm_max_output_tokens: int = 2000

    # hard skip (for tests / dev)
    skip_llm: bool = False

    # JSON-only mode is for retries only, not initial generation
    llm_json_only: bool = False

    # helpful during debugging
    llm_verbose: bool = False


@dataclass
class PipelineResult:
    df_gpx: pd.DataFrame

    df_res: pd.DataFrame
    seg: Any
    key_climbs: pd.DataFrame
    course_summary: Dict[str, Any]
    segment_summaries: list[str]
    climb_summaries: list[str]

    strategy_text: str
    strategy_data: Dict[str, Any]

    overview_df: pd.DataFrame
    segments_df: pd.DataFrame
    climbs_df: pd.DataFrame
    strategy_tables: Dict[str, pd.DataFrame]


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """
    One canonical end-to-end runner.
    This is what the notebook / web app / CLI should call.
    """

    # 1) Load + clean GPX (robust elevation handling)
    df_gpx = load_gpx_to_df(
        cfg.gpx_path,
        window_length=cfg.savgol_window_length,
        polyorder=cfg.savgol_polyorder,
    )

    # 2) Build full course model
    df_res, seg, key_climbs, course_summary, segment_summaries, climb_summaries = (
        build_full_course_model(df_gpx)
    )

    # Make elevation diagnostics available downstream
    if isinstance(course_summary, dict) and "elevation_quality" not in course_summary:
        eq = getattr(df_gpx, "attrs", {}).get("elevation_quality")
        if eq is not None:
            course_summary["elevation_quality"] = eq

    # 3) Strategy (LLM or skipped)
    strategy_text = ""
    strategy_data: Dict[str, Any] = {}
    strategy_tables: Dict[str, pd.DataFrame] = {}

    should_skip = bool(cfg.skip_llm) or (str(cfg.llm_model).upper() == "SKIP")

    if not should_skip:
        athlete_profile = get_default_athlete_profile()
        strategy_text, strategy_data = generate_race_strategy(
            course_summary=course_summary,
            segment_summaries=segment_summaries,
            climb_summaries=climb_summaries,
            athlete_profile=athlete_profile,
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