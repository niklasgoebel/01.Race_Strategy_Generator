from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start,end) runs where mask is True."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    return runs


def _rolling_median_and_mad(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling robust stats using pandas (simple + reliable).
    Returns median and MAD (median absolute deviation).
    """
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median().to_numpy()
    abs_dev = np.abs(x - med)
    mad = pd.Series(abs_dev).rolling(window, center=True, min_periods=1).median().to_numpy()
    return med, mad


def clean_elevation(
    df: pd.DataFrame,
    *,
    elev_col: str,
    dist_col: str,
    # --- spike detection (keeps backwards compat with your tests) ---
    spike_z_thresh: float = 6.0,
    spike_window_points: int = 11,
    # --- long zero-run dropout handling ---
    zero_floor_m: float = 0.0,
    min_zero_run_m: float = 50.0,
    # --- smoothing ---
    savgol_window_length: int = 13,
    savgol_polyorder: int = 3,
    apply_savgol: bool = True,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Robust elevation cleaning for GPX.

    Handles:
    - Missing values -> interpolate
    - Single-point spikes (robust z-score vs rolling median/MAD) -> set NaN + interpolate
    - Long runs at 0m (device dropout) -> set NaN + interpolate
    - Optional Savitzky–Golay smoothing (after repairs)

    Returns:
      cleaned elevation as pd.Series (so tests can call .to_numpy())
      quality dict
    """
    elev_raw = pd.to_numeric(df[elev_col], errors="coerce").astype(float).to_numpy()
    dist = pd.to_numeric(df[dist_col], errors="coerce").astype(float).to_numpy()

    n = len(elev_raw)
    if n == 0:
        raise ValueError("Empty elevation series")

    # Start series
    s = pd.Series(elev_raw, index=df.index)

    # 1) Missing interpolation
    missing_mask = ~np.isfinite(s.to_numpy(dtype=float))
    missing_frac = float(missing_mask.mean())
    if missing_mask.any():
        s = s.interpolate(limit_direction="both")

    # 2) Spike detection (single-point / short spikes)
    x = s.to_numpy(dtype=float)
    med, mad = _rolling_median_and_mad(x, window=int(spike_window_points))
    eps = 1e-9
    z = np.abs(x - med) / (1.4826 * mad + eps)  # robust z-score

    # Mark spikes (exclude NaNs)
    spike_mask = np.isfinite(x) & (z > float(spike_z_thresh))

    spikes_fixed = int(spike_mask.sum())
    if spikes_fixed > 0:
        s.loc[spike_mask] = np.nan
        s = s.interpolate(limit_direction="both")

    # 3) Long 0m dropout runs -> treat as missing and interpolate
    x2 = s.to_numpy(dtype=float)
    zeroish = np.isfinite(x2) & (x2 <= float(zero_floor_m))
    runs = _find_runs(zeroish)

    zeros_fixed = 0
    if runs:
        for a, b in runs:
            run_m = float(dist[b] - dist[a]) if b > a else 0.0
            if run_m >= float(min_zero_run_m):
                s.iloc[a : b + 1] = np.nan
                zeros_fixed += (b - a + 1)

        if zeros_fixed > 0:
            s = s.interpolate(limit_direction="both")

    # 4) Optional SavGol smoothing
    used_savgol = False
    x3 = s.to_numpy(dtype=float)

    if apply_savgol and savgol_filter is not None and n >= 7:
        wl = int(savgol_window_length)
        if wl % 2 == 0:
            wl += 1
        wl = min(wl, n if n % 2 == 1 else n - 1)

        if wl >= 5 and wl > savgol_polyorder:
            x3 = savgol_filter(x3, window_length=wl, polyorder=int(savgol_polyorder))
            used_savgol = True

    cleaned = pd.Series(x3, index=df.index)

    # Diagnostics
    total_fixed = spikes_fixed + zeros_fixed
    spike_frac = float(total_fixed / max(n, 1))

    notes = []
    if spikes_fixed > 0:
        notes.append("spike_outliers_fixed")
    if zeros_fixed > 0:
        notes.append("zero_run_fixed")
    if missing_mask.any():
        notes.append("missing_interp")
    if used_savgol:
        notes.append("savgol")
    if not notes:
        notes.append("ok")

    quality = {
        "missing_frac": float(missing_frac),
        "spikes_fixed": int(total_fixed),
        "spike_frac": float(spike_frac),
        "used_savgol": bool(used_savgol),
        "notes": ";".join(notes),
    }

    return cleaned, quality