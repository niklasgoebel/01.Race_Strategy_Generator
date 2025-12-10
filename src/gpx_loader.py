# src/gpx_loader.py

from __future__ import annotations

from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Any

import gpxpy
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance between two lat/lon points in meters.
    """
    R = 6371000.0  # Earth radius in meters

    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

    return R * c


def _parse_gpx_to_points(gpx_file_path: str) -> pd.DataFrame:
    """
    Read GPX and return raw points as a DataFrame with
    columns: ['lat', 'lon', 'elev', 'time'].

    IMPORTANT: keep the original track order (no sorting).
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

    df = pd.DataFrame(points)

    # Do NOT sort here – we trust the GPX order
    df = df.reset_index(drop=True)

    return df


def _compute_step_and_cum_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute step_distance and cum_distance in meters using haversine.
    """
    step_distances = [0.0]

    for i in range(1, len(df)):
        d = haversine_m(
            df.loc[i - 1, "lat"],
            df.loc[i - 1, "lon"],
            df.loc[i, "lat"],
            df.loc[i, "lon"],
        )
        step_distances.append(d)

    df = df.copy()
    df["step_distance"] = step_distances
    df["cum_distance"] = df["step_distance"].cumsum()

    return df


def _clean_and_smooth_elevation(
    df: pd.DataFrame,
    elevation_floor_m: float = 200.0,
    window_length: int = 13,
    polyorder: int = 3,
) -> pd.DataFrame:
    """
    Apply the SAME elevation pipeline we used in the notebook:

      1. Start from elev_raw.
      2. Set all values below elevation_floor_m to NaN.
      3. Interpolate + bfill + ffill.
      4. Apply Savitzky–Golay smoothing to get elev_smooth.

    Assumes df has 'elev_raw' column.
    """
    df = df.copy()

    elev_clean = df["elev_raw"].copy()

    # 1) absolute floor via NaN
    elev_clean[elev_clean < elevation_floor_m] = np.nan

    # 2) interpolate across NaNs and fill ends
    elev_clean = elev_clean.interpolate().bfill().ffill()

    df["elev_clean"] = elev_clean

    # 3) Savitzky–Golay smoothing (same params we validated before)
    if len(df) >= window_length and window_length % 2 == 1:
        df["elev_smooth"] = savgol_filter(
            df["elev_clean"].values,
            window_length=window_length,
            polyorder=polyorder,
        )
    else:
        df["elev_smooth"] = df["elev_clean"]

    return df


def load_gpx_to_df(
    gpx_file_path: str,
    elevation_floor_m: float = 200.0,
    window_length: int = 13,
    polyorder: int = 3,
) -> pd.DataFrame:
    """
    High-level loader used by the rest of the project.

    Steps:
      1. Parse GPX into lat/lon/elev/time.
      2. Compute step_distance + cum_distance in meters.
      3. Apply elevation floor + Savitzky–Golay smoothing.

    Returns a DataFrame with at least:
      - lat, lon, time
      - elev_raw, elev_clean, elev_smooth
      - step_distance, cum_distance (meters)
    """
    df = _parse_gpx_to_points(gpx_file_path)

    # Keep original elevation
    df["elev_raw"] = df["elev"]

    # Distance in meters
    df = _compute_step_and_cum_distance(df)

    # Elevation cleaning / smoothing
    df = _clean_and_smooth_elevation(
        df,
        elevation_floor_m=elevation_floor_m,
        window_length=window_length,
        polyorder=polyorder,
    )

    return df