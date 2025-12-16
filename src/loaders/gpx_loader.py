from __future__ import annotations

from typing import Any, Dict, List

import gpxpy
import numpy as np
import pandas as pd

from src.elevation import clean_elevation
from src.utils.geo import compute_step_and_cum_distance, ensure_strictly_increasing


def _parse_gpx_to_points(gpx_file_path: str) -> pd.DataFrame:
    """
    Parse GPX into a DataFrame with columns:
      - lat, lon, elev, time

    Keeps original GPX order (no sorting).
    """
    with open(gpx_file_path, "r") as f:
        gpx = gpxpy.parse(f)

    points: List[Dict[str, Any]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(
                    {
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "elev": float(point.elevation) if point.elevation is not None else np.nan,
                        "time": point.time,
                    }
                )

    return pd.DataFrame(points).reset_index(drop=True)


def load_gpx_to_df(
    gpx_file_path: str,
    window_length: int = None,
    polyorder: int = None,
) -> pd.DataFrame:
    """
    High-level GPX → DataFrame loader.

    Returns df with:
      - lat, lon, time
      - elev_raw, elev_smooth
      - step_distance, cum_distance (meters)

    Also attaches df.attrs["elevation_quality"] with cleaning diagnostics.
    
    Note: Smoothing parameters are automatically determined based on data characteristics
    unless explicitly provided.
    """
    df = _parse_gpx_to_points(gpx_file_path)

    # --- raw elevation ---
    df["elev_raw"] = pd.to_numeric(df["elev"], errors="coerce").astype(float)

    # --- distance (meters) ---
    df = compute_step_and_cum_distance(df)
    df["cum_distance"] = ensure_strictly_increasing(
        pd.to_numeric(df["cum_distance"], errors="coerce").astype(float)
    )

    # --- elevation cleaning (robust: spikes, zero runs, missing) ---
    # Use adaptive smoothing by default (auto-determines parameters from data characteristics)
    use_auto_smoothing = (window_length is None or polyorder is None)
    
    elev_smooth, quality = clean_elevation(
        df,
        elev_col="elev_raw",
        dist_col="cum_distance",
        savgol_window_length=int(window_length) if window_length is not None else None,
        savgol_polyorder=int(polyorder) if polyorder is not None else None,
        apply_savgol=True,
        auto_smoothing=use_auto_smoothing,
    )

    # IMPORTANT: ensure alignment + numeric dtype
    df["elev_smooth"] = (
        pd.Series(elev_smooth, index=df.index)
        .astype(float)
    )

    # --- attach diagnostics for downstream UI / prompting ---
    df.attrs["elevation_quality"] = quality

    return df