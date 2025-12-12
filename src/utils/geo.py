# src/utils/geo.py

from __future__ import annotations

from math import radians, sin, cos, sqrt, atan2
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in meters."""
    R = 6371000.0  # meters

    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

    return R * c


def compute_step_and_cum_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - step_distance (m)
      - cum_distance (m)
    Requires columns: lat, lon
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

    out = df.copy()
    out["step_distance"] = np.array(step_distances, dtype=float)
    out["cum_distance"] = out["step_distance"].cumsum()
    return out


def apply_elevation_floor_interpolate(
    elev: pd.Series,
    elevation_floor_m: float = 200.0,
) -> pd.Series:
    """
    Our proven 'valley remover':
      - set elevations below a floor to NaN
      - interpolate + bfill + ffill
    """
    e = elev.astype(float).copy()
    e[e < elevation_floor_m] = np.nan
    e = e.interpolate().bfill().ffill()
    return e


def savgol_smooth(
    values: pd.Series,
    window_length: int = 13,
    polyorder: int = 3,
) -> pd.Series:
    """
    Savitzky–Golay smoothing. If too few points or invalid window, returns original.
    """
    x = values.astype(float).to_numpy()

    if len(x) < window_length or window_length % 2 == 0:
        return pd.Series(values.values, index=values.index)

    smoothed = savgol_filter(x, window_length=window_length, polyorder=polyorder)
    return pd.Series(smoothed, index=values.index)