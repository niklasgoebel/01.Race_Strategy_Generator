# src/utils/geo.py

from __future__ import annotations

from math import radians, sin, cos, sqrt, atan2
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in meters."""
    R = 6_371_000.0

    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2.0 * R * atan2(sqrt(a), sqrt(1.0 - a))


def compute_step_and_cum_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Adds step_distance (m) and cum_distance (m)."""
    step_distances = [0.0]
    for i in range(1, len(df)):
        step_distances.append(
            haversine_m(
                df.loc[i - 1, "lat"],
                df.loc[i - 1, "lon"],
                df.loc[i, "lat"],
                df.loc[i, "lon"],
            )
        )

    out = df.copy()
    out["step_distance"] = np.asarray(step_distances, dtype=float)
    out["cum_distance"] = out["step_distance"].cumsum()
    return out


def ensure_strictly_increasing(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Ensure monotonic increasing values (prevents gradient division by zero)."""
    x = series.to_numpy(dtype=float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return pd.Series(x, index=series.index, name=series.name)


def apply_elevation_floor_interpolate(
    elev: pd.Series,
    elevation_floor_m: float = 200.0,
) -> pd.Series:
    """Remove spurious valleys via absolute elevation floor."""
    e = elev.astype(float).copy()
    e[e < elevation_floor_m] = np.nan
    return e.interpolate().bfill().ffill()


def savgol_smooth(
    values: pd.Series,
    window_length: int = 13,
    polyorder: int = 3,
) -> pd.Series:
    """Savitzky–Golay smoothing."""
    x = values.astype(float).to_numpy()

    if len(x) < window_length or window_length % 2 == 0:
        return pd.Series(values.values, index=values.index)

    return pd.Series(
        savgol_filter(x, window_length=window_length, polyorder=polyorder),
        index=values.index,
    )