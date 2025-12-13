# src/course_model.py

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_uniform(df_gpx: pd.DataFrame, step_m: float = 20.0) -> pd.DataFrame:
    """
    Assumes df_gpx has:
      - 'cum_distance' in meters
      - 'lat', 'lon'
      - 'elev_smooth' (already cleaned & smoothed)

    Returns a resampled DataFrame every `step_m` meters.
    """
    dist = df_gpx["cum_distance"].to_numpy()
    lat = df_gpx["lat"].to_numpy()
    lon = df_gpx["lon"].to_numpy()
    elev = df_gpx["elev_smooth"].to_numpy()

    total = dist[-1]

    # Include endpoint so the final distance is represented
    target = np.arange(0, total + step_m, step_m)
    target = target[target <= total]

    lat_u = np.interp(target, dist, lat)
    lon_u = np.interp(target, dist, lon)
    elev_u = np.interp(target, dist, elev)

    return pd.DataFrame(
        {
            "cum_distance": target,
            "lat": lat_u,
            "lon": lon_u,
            "elev_smooth": elev_u,
        }
    )


# ---------------------------------------------------------------------------
# Gradient + segmentation
# ---------------------------------------------------------------------------

def compute_window_gradient(df_res: pd.DataFrame, window_m: float = 100.0) -> pd.DataFrame:
    dist = df_res["cum_distance"].to_numpy()
    elev = df_res["elev_smooth"].to_numpy()

    grad = np.zeros_like(elev, dtype=float)

    for i in range(len(dist)):
        target = dist[i] + window_m
        j = np.searchsorted(dist, target)
        if j >= len(dist):
            j = len(dist) - 1

        run = dist[j] - dist[i]
        rise = elev[j] - elev[i]
        grad[i] = (rise / run) * 100 if run > 0 else np.nan

    # Median smoothing to stabilize classification
    df_res["gradient_final"] = (
        pd.Series(grad).rolling(25, center=True, min_periods=1).median()
    )
    return df_res


def classify_gradient(g: float) -> str:
    if g > 8:
        return "climb_steep"
    elif g > 3:
        return "climb_moderate"
    elif g < -8:
        return "descent_steep"
    elif g < -3:
        return "descent_moderate"
    else:
        return "runnable"


def add_segment_labels(df_res: pd.DataFrame) -> pd.DataFrame:
    df_res["segment_type_raw"] = df_res["gradient_final"].apply(classify_gradient)
    return df_res


def segments_from_labels(df_res: pd.DataFrame) -> pd.DataFrame:
    labels = df_res["segment_type_raw"].to_numpy()

    segments = []
    current = None
    start = 0

    for i, lab in enumerate(labels):
        if current is None:
            current = lab
            start = i
        elif lab != current:
            segments.append((start, i - 1, current))
            current = lab
            start = i

    segments.append((start, len(labels) - 1, current))

    rows = []
    for start, end, typ in segments:
        dist = df_res["cum_distance"]
        elev = df_res["elev_smooth"]
        g = df_res["gradient_final"]

        d0, d1 = dist.iloc[start], dist.iloc[end]

        rows.append(
            {
                "type": typ,
                "start_km": d0 / 1000.0,
                "end_km": d1 / 1000.0,
                "distance_km": (d1 - d0) / 1000.0,
                "elev_change_m": elev.iloc[end] - elev.iloc[start],
                "avg_gradient": g.iloc[start : end + 1].mean(),
                "points": end - start + 1,
            }
        )

    return pd.DataFrame(rows)


def merge_micro_segments(
    segments_df: pd.DataFrame,
    df_res: pd.DataFrame,
    min_segment_km: float = 0.2,
) -> pd.DataFrame:
    if segments_df.empty:
        return segments_df.copy()

    merged = []
    current = segments_df.iloc[0].copy()

    cum = df_res["cum_distance"].to_numpy()
    elev = df_res["elev_smooth"].to_numpy()
    g = df_res["gradient_final"].to_numpy()

    for i in range(1, len(segments_df)):
        row = segments_df.iloc[i]

        if row["distance_km"] < min_segment_km:
            current["end_km"] = row["end_km"]
            current["distance_km"] = current["end_km"] - current["start_km"]

            start_m = float(current["start_km"]) * 1000.0
            end_m = float(current["end_km"]) * 1000.0

            start_idx = int(np.searchsorted(cum, start_m, side="left"))
            end_idx = int(np.searchsorted(cum, end_m, side="right"))

            start_idx = max(0, min(start_idx, len(cum) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(cum)))

            current["elev_change_m"] = float(elev[end_idx - 1] - elev[start_idx])
            current["avg_gradient"] = float(np.nanmean(g[start_idx:end_idx]))
        else:
            merged.append(current.copy())
            current = row.copy()

    merged.append(current.copy())
    return pd.DataFrame(merged)


# ---------------------------------------------------------------------------
# Enrichment + summaries
# ---------------------------------------------------------------------------

def enrich_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    seg = segments_df.copy()
    seg["gain_m"] = seg["elev_change_m"].clip(lower=0)
    seg["loss_m"] = (-seg["elev_change_m"]).clip(lower=0)
    seg["is_climb"] = seg["type"].str.startswith("climb")
    seg["is_descent"] = seg["type"].str.startswith("descent")
    seg["is_runnable"] = seg["type"] == "runnable"
    return seg


def extract_key_climbs(
    seg: pd.DataFrame,
    min_gain_m: float = 50.0,
    min_dist_km: float = 0.4,
    max_avg_gradient: float = 30.0,
) -> pd.DataFrame:
    climbs = seg[
        (seg["is_climb"])
        & (seg["gain_m"] >= min_gain_m)
        & (seg["distance_km"] >= min_dist_km)
        & (seg["avg_gradient"].abs() <= max_avg_gradient)
    ].copy()

    if climbs.empty:
        return climbs

    climbs = climbs.sort_values(
        ["gain_m", "avg_gradient"], ascending=[False, False]
    )
    climbs["rank_by_gain"] = (
        climbs["gain_m"].rank(ascending=False, method="dense").astype(int)
    )
    return climbs


def summarize_course_overview(
    df_res: pd.DataFrame,
    seg: pd.DataFrame,
    elevation_quality: dict | None = None,
) -> dict:
    dist_km = df_res["cum_distance"].iloc[-1] / 1000.0

    elev = df_res["elev_smooth"].to_numpy()
    diffs = np.diff(elev)

    total_gain = diffs[diffs > 0].sum()
    total_loss = -diffs[diffs < 0].sum()

    summary = {
        "total_distance_km": round(float(dist_km), 1),
        "total_gain_m": int(total_gain),
        "total_loss_m": int(total_loss),
        "num_segments": int(len(seg)),
    }

    if elevation_quality is not None:
        summary["elevation_quality"] = elevation_quality

    return summary


def summarize_segments(seg: pd.DataFrame) -> list[str]:
    out = []
    for _, row in seg.iterrows():
        out.append(
            f"{row['type']} | {row['start_km']:.1f}–{row['end_km']:.1f} km | "
            f"{row['distance_km']:.1f} km | gain {row['gain_m']:.0f} m | "
            f"loss {row['loss_m']:.0f} m | avg gradient {row['avg_gradient']:.1f}%"
        )
    return out


def summarize_key_climbs(key_climbs: pd.DataFrame) -> list[str]:
    out = []
    for _, row in key_climbs.iterrows():
        out.append(
            f"Climb #{row['rank_by_gain']} — "
            f"{row['start_km']:.1f}–{row['end_km']:.1f} km "
            f"({row['distance_km']:.1f} km), gain {row['gain_m']:.0f} m, "
            f"avg gradient {row['avg_gradient']:.1f}%"
        )
    return out


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------

def build_full_course_model(
    df_gpx: pd.DataFrame,
    step_m: int = 20,
    window_m: int = 100,
    min_segment_km: float = 0.2,
):
    """
    Assumes df_gpx already has:
      - cum_distance (m)
      - elev_smooth (robustly cleaned)
    """
    df_res = resample_to_uniform(df_gpx, step_m=step_m)
    df_res = compute_window_gradient(df_res, window_m=window_m)
    df_res = add_segment_labels(df_res)

    raw_segments = segments_from_labels(df_res)
    merged_segments = merge_micro_segments(
        raw_segments, df_res, min_segment_km=min_segment_km
    )

    seg = enrich_segments(merged_segments)
    key_climbs = extract_key_climbs(seg)

    elevation_quality = (
        df_gpx.attrs.get("elevation_quality") if hasattr(df_gpx, "attrs") else None
    )

    course_summary = summarize_course_overview(
        df_res, seg, elevation_quality=elevation_quality
    )
    segment_summaries = summarize_segments(seg)
    climb_summaries = summarize_key_climbs(key_climbs)

    return df_res, seg, key_climbs, course_summary, segment_summaries, climb_summaries