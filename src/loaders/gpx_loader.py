# src/loaders/gpx_loader.py

from __future__ import annotations

from typing import List, Dict, Any

import gpxpy
import numpy as np
import pandas as pd

from src.utils.geo import (
    compute_step_and_cum_distance,
    apply_elevation_floor_interpolate,
    savgol_smooth,
    ensure_strictly_increasing,
)

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
    elevation_floor_m: float = 200.0,
    window_length: int = 13,
    polyorder: int = 3,
) -> pd.DataFrame:
    """
    High-level GPX → DataFrame loader.

    Returns df with:
      - lat, lon, time
      - elev_raw, elev_clean, elev_smooth
      - step_distance, cum_distance (meters)
    """
    df = _parse_gpx_to_points(gpx_file_path)

    df["elev_raw"] = df["elev"].astype(float)

    # Distance (meters)
    df = compute_step_and_cum_distance(df)
    df["cum_distance"] = ensure_strictly_increasing(df["cum_distance"])

    # Elevation cleaning
    df["elev_clean"] = apply_elevation_floor_interpolate(df["elev_raw"], elevation_floor_m=elevation_floor_m)
    df["elev_smooth"] = savgol_smooth(df["elev_clean"], window_length=window_length, polyorder=polyorder)

    return df