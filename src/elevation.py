from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None


def validate_elevation_metrics(
    total_gain_m: float,
    total_loss_m: float,
    distance_km: float,
    course_type: str = "trail"
) -> Dict[str, any]:
    """
    Validate elevation metrics and flag potential data quality issues.
    
    Checks:
    - Gain/loss per km against expected ranges
    - Gain/loss balance (should be similar for loop courses)
    - Extreme values that suggest GPS drift or data corruption
    
    Args:
        total_gain_m: Total elevation gain in meters
        total_loss_m: Total elevation loss in meters
        distance_km: Total distance in kilometers
        course_type: "road", "trail", or "mountain"
    
    Returns:
        Dictionary with validation results:
        - gain_per_km: Average gain per kilometer
        - loss_per_km: Average loss per kilometer
        - quality: "good", "check_needed", or "poor"
        - warnings: List of warning messages
        - flags: List of specific issues detected
    """
    if distance_km <= 0:
        return {
            "quality": "poor",
            "warnings": ["Invalid distance (≤0 km)"],
            "flags": ["invalid_distance"]
        }
    
    gain_per_km = total_gain_m / distance_km
    loss_per_km = total_loss_m / distance_km
    
    # Expected ranges by course type (conservative bounds)
    ranges = {
        "road": (0, 35),       # Road races: 0-35m gain/km
        "trail": (15, 100),    # Trail races: 15-100m gain/km
        "mountain": (40, 200)  # Mountain ultras: 40-200m gain/km
    }
    
    min_expected, max_expected = ranges.get(course_type, (0, 200))
    
    warnings: List[str] = []
    flags: List[str] = []
    
    # Check gain per km
    if gain_per_km > max_expected:
        warnings.append(
            f"Elevation gain ({gain_per_km:.0f}m/km) is unusually high for {course_type} course. "
            f"Expected: {min_expected}-{max_expected}m/km. "
            f"Possible GPS drift or data quality issues."
        )
        flags.append("high_gain_per_km")
    
    if gain_per_km < min_expected and course_type != "road":
        warnings.append(
            f"Elevation gain ({gain_per_km:.0f}m/km) is unusually low for {course_type} course. "
            f"Expected: {min_expected}-{max_expected}m/km. "
            f"Check GPX data quality or course_type setting."
        )
        flags.append("low_gain_per_km")
    
    # Check gain/loss balance
    imbalance = abs(total_gain_m - total_loss_m)
    imbalance_pct = (imbalance / max(total_gain_m, total_loss_m, 1)) * 100
    
    if imbalance > 300 or imbalance_pct > 15:
        warnings.append(
            f"Gain/loss imbalance: {imbalance:.0f}m difference ({imbalance_pct:.0f}%). "
            f"This suggests either: (1) point-to-point course with net elevation change, "
            f"or (2) GPS data quality issues."
        )
        flags.append("gain_loss_imbalance")
    
    # Check for extreme values
    if total_gain_m > distance_km * 300:
        warnings.append(
            f"Extreme elevation gain: {total_gain_m:.0f}m over {distance_km:.1f}km. "
            f"This is extremely unlikely and suggests severe GPS drift."
        )
        flags.append("extreme_gain")
    
    if total_gain_m < 10 and distance_km > 10 and course_type != "road":
        warnings.append(
            f"Very low elevation gain: {total_gain_m:.0f}m over {distance_km:.1f}km. "
            f"GPX may be missing elevation data or course is flatter than expected."
        )
        flags.append("minimal_gain")
    
    # Determine overall quality
    if not warnings:
        quality = "good"
    elif len(warnings) == 1 and "imbalance" in flags[0]:
        quality = "check_needed"  # Imbalance alone might be OK (point-to-point)
    elif any(f in flags for f in ["extreme_gain", "high_gain_per_km", "minimal_gain"]):
        quality = "poor"
    else:
        quality = "check_needed"
    
    return {
        "gain_per_km": round(gain_per_km, 1),
        "loss_per_km": round(loss_per_km, 1),
        "imbalance_m": round(imbalance, 0),
        "imbalance_pct": round(imbalance_pct, 1),
        "quality": quality,
        "warnings": warnings,
        "flags": flags,
        "course_type": course_type,
    }


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


def adaptive_smoothing_params(
    df: pd.DataFrame,
    dist_col: str,
    total_distance_km: float = None
) -> Dict[str, int]:
    """
    Determine optimal Savitzky-Golay smoothing parameters based on data characteristics.
    
    Considers:
    - Point density (high res GPX needs more smoothing)
    - Data quality (noisy data needs more smoothing)
    
    Args:
        df: Dataframe with GPX data
        dist_col: Column name for distance
        total_distance_km: Total distance in km (computed if not provided)
    
    Returns:
        Dict with 'window_length' and 'polyorder' keys
    """
    if total_distance_km is None:
        dist = pd.to_numeric(df[dist_col], errors="coerce").astype(float)
        total_distance_km = float(dist.max() - dist.min()) / 1000.0
    
    # Calculate point density
    points_per_km = len(df) / max(total_distance_km, 0.1)
    
    # Determine window size based on point density
    # High resolution (>200 pts/km, e.g., 1pt/5m) needs more smoothing
    # Low resolution (<50 pts/km, e.g., 1pt/20m+) needs less smoothing
    if points_per_km > 200:
        # Very high resolution (e.g., 1 point per second while moving)
        window_length = 21
        polyorder = 3
    elif points_per_km > 100:
        # High resolution (e.g., 1 point every 10m)
        window_length = 15
        polyorder = 3
    elif points_per_km > 50:
        # Medium resolution (standard GPX)
        window_length = 13
        polyorder = 3
    else:
        # Low resolution (pre-processed or sparse GPX)
        window_length = 9
        polyorder = 2
    
    # Ensure window_length is odd and >= polyorder + 2
    window_length = max(window_length, polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1
    
    return {
        "window_length": window_length,
        "polyorder": polyorder,
        "points_per_km": points_per_km
    }


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
    savgol_window_length: int = None,
    savgol_polyorder: int = None,
    apply_savgol: bool = True,
    auto_smoothing: bool = False,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Robust elevation cleaning for GPX.

    Handles:
    - Missing values -> interpolate
    - Single-point spikes (robust z-score vs rolling median/MAD) -> set NaN + interpolate
    - Long runs at 0m (device dropout) -> set NaN + interpolate
    - Optional Savitzky–Golay smoothing (after repairs)
    
    Args:
        auto_smoothing: If True, automatically determine smoothing params based on data
        savgol_window_length: Manual window length (ignored if auto_smoothing=True)
        savgol_polyorder: Manual polyorder (ignored if auto_smoothing=True)

    Returns:
      cleaned elevation as pd.Series (so tests can call .to_numpy())
      quality dict (includes 'smoothing_params' if auto_smoothing used)
    """
    # Auto-determine smoothing parameters if requested
    if auto_smoothing and apply_savgol:
        params = adaptive_smoothing_params(df, dist_col)
        savgol_window_length = params["window_length"]
        savgol_polyorder = params["polyorder"]
        smoothing_method = "adaptive"
    else:
        # Use defaults if not specified
        if savgol_window_length is None:
            savgol_window_length = 13
        if savgol_polyorder is None:
            savgol_polyorder = 3
        smoothing_method = "manual"
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
        "smoothing_method": smoothing_method if used_savgol else "none",
        "smoothing_window": int(savgol_window_length) if used_savgol else 0,
        "smoothing_polyorder": int(savgol_polyorder) if used_savgol else 0,
        "notes": ";".join(notes),
    }

    return cleaned, quality