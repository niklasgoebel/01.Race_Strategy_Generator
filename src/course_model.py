# src/course_model.py

import numpy as np
import pandas as pd

from src.effort_blocks import build_climb_blocks

# -------------------------
# Resampling + gradient
# -------------------------
def resample_to_uniform(df_gpx: pd.DataFrame, step_m: float = 20.0) -> pd.DataFrame:
    """
    Assumes df_gpx has:
      - 'cum_distance' in meters
      - 'lat', 'lon'
      - 'elev_smooth' (already cleaned & smoothed)

    Returns a resampled DataFrame every `step_m` meters.
    """
    dist = df_gpx["cum_distance"].to_numpy(dtype=float)
    lat = df_gpx["lat"].to_numpy(dtype=float)
    lon = df_gpx["lon"].to_numpy(dtype=float)
    elev = df_gpx["elev_smooth"].to_numpy(dtype=float)

    total = float(dist[-1])
    target = np.arange(0.0, total + step_m, step_m)
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


def compute_window_gradient(df_res: pd.DataFrame, window_m: float = 100.0) -> pd.DataFrame:
    dist = df_res["cum_distance"].to_numpy(dtype=float)
    elev = df_res["elev_smooth"].to_numpy(dtype=float)
    grad = np.zeros_like(elev, dtype=float)

    for i in range(len(dist)):
        target = dist[i] + window_m
        j = np.searchsorted(dist, target)
        if j >= len(dist):
            j = len(dist) - 1
        run = dist[j] - dist[i]
        rise = elev[j] - elev[i]
        grad[i] = (rise / run) * 100.0 if run > 0 else np.nan

    # median filter to reduce jitter
    df_res["gradient_final"] = (
        pd.Series(grad).rolling(25, center=True, min_periods=1).median()
    )
    return df_res


# -------------------------
# Labeling (3-class: climb/descent/runnable)
# -------------------------
def classify_gradient(g: float) -> str:
    if g > 3:
        return "climb"
    elif g < -3:
        return "descent"
    else:
        return "runnable"


def add_segment_labels(df_res: pd.DataFrame) -> pd.DataFrame:
    df_res["segment_type_raw"] = df_res["gradient_final"].apply(classify_gradient)
    return df_res


# -------------------------
# Label smoothing (debounce by distance)
# -------------------------
def _smooth_point_labels_by_distance(
    df_res: pd.DataFrame,
    *,
    label_col: str = "segment_type_raw",
    dist_col: str = "cum_distance",
    min_run_m: float = 200.0,
) -> pd.DataFrame:
    """
    Replace very short label runs (by distance) with the surrounding label.

    Example: runnable for 60m sandwiched between climbs -> treat as noise.
    """
    labels = df_res[label_col].to_numpy()
    dist = df_res[dist_col].to_numpy(dtype=float)

    if len(labels) < 3:
        df_res["segment_type_smooth"] = labels
        return df_res

    runs = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = labels[i]
    runs.append((start, len(labels) - 1, cur))

    out = labels.copy()

    for r_i, (a, b, lab) in enumerate(runs):
        if a == 0 or b == len(labels) - 1:
            continue

        run_m = float(dist[b] - dist[a]) if b > a else 0.0
        if run_m >= float(min_run_m):
            continue

        prev_lab = runs[r_i - 1][2]
        next_lab = runs[r_i + 1][2]

        if prev_lab == next_lab:
            out[a : b + 1] = prev_lab
        else:
            prev_a, prev_b, _ = runs[r_i - 1]
            next_a, next_b, _ = runs[r_i + 1]
            prev_len = float(dist[prev_b] - dist[prev_a]) if prev_b > prev_a else 0.0
            next_len = float(dist[next_b] - dist[next_a]) if next_b > next_a else 0.0
            out[a : b + 1] = prev_lab if prev_len >= next_len else next_lab

    df_res["segment_type_smooth"] = out
    return df_res


# -------------------------
# Segments from point labels
# -------------------------
def segments_from_labels(df_res: pd.DataFrame, label_col: str) -> pd.DataFrame:
    labels = df_res[label_col].to_numpy()
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

    dist = df_res["cum_distance"]
    elev = df_res["elev_smooth"]
    g = df_res["gradient_final"]

    rows = []
    for s, e, typ in segments:
        d0, d1 = float(dist.iloc[s]), float(dist.iloc[e])
        rows.append(
            {
                "type": typ,
                "start_km": d0 / 1000.0,
                "end_km": d1 / 1000.0,
                "distance_km": (d1 - d0) / 1000.0,
                "elev_change_m": float(elev.iloc[e] - elev.iloc[s]),
                "avg_gradient": float(np.nanmean(g.iloc[s : e + 1])),
                "points": int(e - s + 1),
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# Merge short segments (postprocessing)
# -------------------------
def _merge_short_segments(seg_df: pd.DataFrame, *, min_segment_km: float = 0.4) -> pd.DataFrame:
    """
    Merge segments shorter than min_segment_km into a neighbor.
    Preference: merge into neighbor with same category (climb/descent/runnable),
    otherwise merge into larger neighbor by distance.
    """
    if seg_df.empty or len(seg_df) <= 1:
        return seg_df.copy()

    seg = seg_df.reset_index(drop=True).copy()

    def cat(t: str) -> str:
        # already 3-class, but keep helper for safety
        if t == "climb":
            return "climb"
        if t == "descent":
            return "descent"
        return "runnable"

    i = 0
    while i < len(seg):
        if seg.loc[i, "distance_km"] >= min_segment_km or len(seg) <= 1:
            i += 1
            continue

        left = i - 1 if i - 1 >= 0 else None
        right = i + 1 if i + 1 < len(seg) else None

        if left is None:
            target = right
        elif right is None:
            target = left
        else:
            c = cat(str(seg.loc[i, "type"]))
            cl = cat(str(seg.loc[left, "type"]))
            cr = cat(str(seg.loc[right, "type"]))

            if cl == c and cr != c:
                target = left
            elif cr == c and cl != c:
                target = right
            else:
                target = (
                    left
                    if seg.loc[left, "distance_km"] >= seg.loc[right, "distance_km"]
                    else right
                )

        a = min(i, target)
        b = max(i, target)

        start_km = float(seg.loc[a, "start_km"])
        end_km = float(seg.loc[b, "end_km"])
        dist_km = float(end_km - start_km)

        # recompute elev change as sum of the two pieces (good approximation post-resample)
        elev_change_m = float(seg.loc[a, "elev_change_m"] + seg.loc[b, "elev_change_m"])

        # weighted avg gradient
        d1 = float(seg.loc[a, "distance_km"])
        d2 = float(seg.loc[b, "distance_km"])
        g1 = float(seg.loc[a, "avg_gradient"])
        g2 = float(seg.loc[b, "avg_gradient"])
        avg_g = (g1 * d1 + g2 * d2) / max(d1 + d2, 1e-9)

        merged = seg.loc[a].copy()
        merged["start_km"] = start_km
        merged["end_km"] = end_km
        merged["distance_km"] = dist_km
        merged["elev_change_m"] = elev_change_m
        merged["avg_gradient"] = float(avg_g)
        merged["points"] = int(seg.loc[a, "points"] + seg.loc[b, "points"])

        # type = dominant by distance
        merged["type"] = seg.loc[a, "type"] if d1 >= d2 else seg.loc[b, "type"]

        seg = pd.concat(
            [seg.iloc[:a], pd.DataFrame([merged]), seg.iloc[b + 1 :]],
            ignore_index=True,
        )

        i = max(0, a - 1)

    return seg


def _coalesce_adjacent_same_type(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee there are no adjacent segments with the same 'type'.
    Fixes cases like runnable + runnable showing separately.
    """
    if seg_df.empty or len(seg_df) <= 1:
        return seg_df.copy()

    seg = seg_df.reset_index(drop=True).copy()
    out = [seg.iloc[0].copy()]

    for i in range(1, len(seg)):
        cur = seg.iloc[i]
        prev = out[-1]

        if cur["type"] == prev["type"]:
            prev_end = float(cur["end_km"])
            prev_start = float(prev["start_km"])
            prev["end_km"] = prev_end
            prev["distance_km"] = float(prev_end - prev_start)
            prev["elev_change_m"] = float(prev["elev_change_m"] + cur["elev_change_m"])
            prev["points"] = int(prev.get("points", 0) + cur.get("points", 0))

            d_prev = float(prev["distance_km"])
            d_cur = float(cur["distance_km"])
            g_prev = float(prev["avg_gradient"])
            g_cur = float(cur["avg_gradient"])
            prev["avg_gradient"] = float((g_prev * d_prev + g_cur * d_cur) / max(d_prev + d_cur, 1e-9))

            out[-1] = prev
        else:
            out.append(cur.copy())

    return pd.DataFrame(out)


# -------------------------
# Enrich + climbs + summaries
# -------------------------
def enrich_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    seg = segments_df.copy()
    seg["gain_m"] = seg["elev_change_m"].clip(lower=0)
    seg["loss_m"] = (-seg["elev_change_m"]).clip(lower=0)
    seg["is_climb"] = seg["type"] == "climb"
    seg["is_descent"] = seg["type"] == "descent"
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

    climbs = climbs.sort_values(["gain_m", "avg_gradient"], ascending=[False, False])
    climbs["rank_by_gain"] = climbs["gain_m"].rank(ascending=False, method="dense").astype(int)
    return climbs


def summarize_course_overview(df_res: pd.DataFrame, seg: pd.DataFrame) -> dict:
    dist_km = float(df_res["cum_distance"].iloc[-1]) / 1000.0
    elev = df_res["elev_smooth"].to_numpy(dtype=float)
    diffs = np.diff(elev)
    total_gain = float(diffs[diffs > 0].sum())
    total_loss = float(-diffs[diffs < 0].sum())

    return {
        "total_distance_km": round(dist_km, 1),
        "total_gain_m": int(total_gain),
        "total_loss_m": int(total_loss),
        "num_segments": int(len(seg)),
    }


def summarize_segments(seg: pd.DataFrame) -> list:
    out = []
    for _, row in seg.iterrows():
        out.append(
            f"{row['type']} | {row['start_km']:.1f}–{row['end_km']:.1f} km | "
            f"{row['distance_km']:.1f} km | gain {row['gain_m']:.0f} m | "
            f"loss {row['loss_m']:.0f} m | avg gradient {row['avg_gradient']:.1f}%"
        )
    return out


def summarize_key_climbs(key_climbs: pd.DataFrame) -> list:
    out = []
    for _, row in key_climbs.iterrows():
        out.append(
            f"Climb #{row['rank_by_gain']} — "
            f"{row['start_km']:.1f}–{row['end_km']:.1f} km "
            f"({row['distance_km']:.1f} km), gain {row['gain_m']:.0f} m, "
            f"avg gradient {row['avg_gradient']:.1f}%"
        )
    return out


# -------------------------
# Main API
# -------------------------
def build_full_course_model(
    df_gpx: pd.DataFrame,
    step_m: int = 20,
    window_m: int = 100,
    min_run_m: float = 200.0,
    min_segment_km: float = 0.4,
):
    """
    Assumes df_gpx already has:
      - cum_distance (m)
      - elev_smooth (cleaned & smoothed)
    """
    df_res = resample_to_uniform(df_gpx, step_m=step_m)
    df_res = compute_window_gradient(df_res, window_m=window_m)
    df_res = add_segment_labels(df_res)

    # smooth noisy label runs before segmenting
    df_res = _smooth_point_labels_by_distance(df_res, min_run_m=min_run_m)

    # Build segments from smoothed labels
    raw_segments = segments_from_labels(df_res, label_col="segment_type_smooth")

    # Merge short segments and ensure no adjacent same-type segments remain
    merged_segments = _merge_short_segments(raw_segments, min_segment_km=min_segment_km)
    merged_segments = _coalesce_adjacent_same_type(merged_segments)

    seg = enrich_segments(merged_segments)
    key_climbs = extract_key_climbs(seg)
    climb_blocks = build_climb_blocks(seg)

    course_summary = summarize_course_overview(df_res, seg)
    segment_summaries = summarize_segments(seg)
    climb_summaries = summarize_key_climbs(key_climbs)

    return df_res, seg, key_climbs, climb_blocks, course_summary, segment_summaries, climb_summaries