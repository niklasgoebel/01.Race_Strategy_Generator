from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

from src.athlete_profile import get_default_athlete_profile
from src.outputs.output_formatter import make_strategy_tables
from src.pipeline import PipelineResult
from src.race_strategy_generator import generate_race_strategy
from src.races.catalog import list_races, load_race_gpx_bytes


# =========================================================
# Cached pipeline runner
# =========================================================
@st.cache_data(show_spinner=False)
def cached_full_run(
    gpx_bytes: bytes,
    filename: str,
    savgol_window_length: int,
    savgol_polyorder: int,
    llm_model: str,
    llm_max_output_tokens: int,
):
    from src.pipeline import PipelineConfig, run_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        gpx_path = Path(tmpdir) / filename
        gpx_path.write_bytes(gpx_bytes)

        cfg = PipelineConfig(
            gpx_path=str(gpx_path),
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder,
            llm_model=llm_model,
            llm_max_output_tokens=llm_max_output_tokens,
        )
        return run_pipeline(cfg)


# =========================================================
# Session state
# =========================================================
def _init_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_source_label", None)
    st.session_state.setdefault("last_file_name", None)

    st.session_state.setdefault("picked_race_id", None)
    st.session_state.setdefault("picked_race_name", None)
    st.session_state.setdefault("picked_race_gpx_name", None)
    st.session_state.setdefault("picked_race_aid_stations", [])  # list[dict{name,km}]


_init_state()


# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Race Strategy Generator", layout="wide")
st.title("🏃‍♂️ Race Strategy Generator")
st.caption("Select a race or upload a GPX file to generate a personalized strategy.")


# =========================================================
# Sidebar — source selection
# =========================================================
st.sidebar.header("Get started")

source_mode = st.sidebar.radio(
    "Choose data source",
    ["Race library (beta)", "Upload GPX"],
    index=0,
)

# -------------------------
# Developer settings (kept)
# -------------------------
with st.sidebar.expander("Developer settings", expanded=False):
    st.caption("Builder settings. Hide/remove in final product.")

    savgol_window = st.number_input(
        "Savgol window length (odd)",
        min_value=5,
        max_value=101,
        value=13,
        step=2,
    )
    savgol_polyorder = st.number_input(
        "Savgol polyorder",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
    )

    llm_model = st.text_input("Model", value="gpt-4.1-mini")
    llm_max_output_tokens = st.slider("Max output tokens", 800, 6000, 3000, 100)

    show_debug = st.checkbox("Show elevation debug overlay", value=False)

# defaults if expander untouched
savgol_window = int(locals().get("savgol_window", 13))
savgol_polyorder = int(locals().get("savgol_polyorder", 3))
llm_model = str(locals().get("llm_model", "gpt-4.1-mini"))
llm_max_output_tokens = int(locals().get("llm_max_output_tokens", 3000))
show_debug = bool(locals().get("show_debug", False))


# =========================================================
# Sidebar — race or GPX input
# =========================================================
gpx_bytes: Optional[bytes] = None
filename: Optional[str] = None
source_label: Optional[str] = None

selected_race = None

if source_mode == "Race library (beta)":
    races = list_races()

    if not races:
        st.sidebar.error("No races found in data/races/catalog.json")
        st.session_state["picked_race_id"] = None
        st.session_state["picked_race_aid_stations"] = []
    else:
        race_names = [r.name for r in races]
        race_by_name = {r.name: r for r in races}

        picked = st.sidebar.selectbox("Select a race", race_names)
        selected_race = race_by_name[picked]

        st.session_state["picked_race_id"] = selected_race.id
        st.session_state["picked_race_name"] = selected_race.name
        st.session_state["picked_race_gpx_name"] = Path(selected_race.gpx_path).name

        # store aid stations as simple dicts (safe for session_state + caching)
        aids = []
        for a in (selected_race.aid_stations or []):
            aids.append({"name": a.name, "km": float(a.km)})
        st.session_state["picked_race_aid_stations"] = aids

else:
    gpx_file = st.sidebar.file_uploader("Upload GPX file", type=["gpx"])
    if gpx_file:
        gpx_bytes = gpx_file.getvalue()
        filename = gpx_file.name
        source_label = "Uploaded GPX"


# =========================================================
# Sidebar actions
# =========================================================
can_run = (
    st.session_state.get("picked_race_id") is not None
    if source_mode == "Race library (beta)"
    else gpx_bytes is not None
)

run_btn = st.sidebar.button("Run analysis", type="primary", disabled=not can_run)

st.sidebar.divider()

retry_btn = st.sidebar.button(
    "Retry strategy only (LLM)",
    disabled=(st.session_state["last_result"] is None),
)

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()


# =========================================================
# Actions
# =========================================================
def _run_full_pipeline(
    gpx_bytes: bytes, filename: str, source_label: str
) -> Tuple[Optional[PipelineResult], Optional[str]]:
    try:
        with st.spinner("Running race analysis..."):
            res = cached_full_run(
                gpx_bytes=gpx_bytes,
                filename=filename,
                savgol_window_length=savgol_window,
                savgol_polyorder=savgol_polyorder,
                llm_model=llm_model,
                llm_max_output_tokens=llm_max_output_tokens,
            )
    except Exception as e:
        return None, str(e)

    st.session_state["last_result"] = res
    st.session_state["last_source_label"] = source_label
    st.session_state["last_file_name"] = filename
    return res, None


def _retry_llm_only() -> Tuple[Optional[PipelineResult], Optional[str]]:
    res: PipelineResult = st.session_state["last_result"]
    athlete_profile = get_default_athlete_profile()

    try:
        with st.spinner("Retrying strategy generation (LLM only)..."):
            strategy_text, strategy_data = generate_race_strategy(
                course_summary=res.course_summary,
                segment_summaries=res.segment_summaries,
                climb_summaries=res.climb_summaries,
                athlete_profile=athlete_profile,
                model=llm_model,
                max_output_tokens=llm_max_output_tokens,
                json_only=True,
            )
    except Exception as e:
        return None, str(e)

    res.strategy_text = strategy_text
    res.strategy_data = strategy_data
    res.strategy_tables = make_strategy_tables(strategy_data) if strategy_data else {}
    return res, None


# =========================================================
# Control flow
# =========================================================
error_msg = None

if run_btn:
    try:
        if source_mode == "Race library (beta)":
            race_id = st.session_state["picked_race_id"]
            race_name = st.session_state["picked_race_name"]
            filename = st.session_state["picked_race_gpx_name"]

            gpx_bytes = load_race_gpx_bytes(race_id)
            source_label = f"Race library: {race_name}"
        else:
            assert gpx_bytes and filename and source_label

        _, error_msg = _run_full_pipeline(gpx_bytes, filename, source_label)

    except Exception as e:
        error_msg = str(e)

if retry_btn:
    _, error_msg = _retry_llm_only()

if error_msg:
    st.error(f"Pipeline failed: {error_msg}")

result: Optional[PipelineResult] = st.session_state["last_result"]
if result is None:
    st.info("Pick a race (or upload GPX) and click **Run analysis**.")
    st.stop()


# =========================================================
# Header summary
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", st.session_state.get("last_source_label", "—"))
c2.metric("File", st.session_state.get("last_file_name", "—"))
c3.metric("Distance", f"{result.course_summary.get('total_distance_km', '—')} km")
c4.metric("Elevation gain", f"{result.course_summary.get('total_gain_m', '—')} m")


# =========================================================
# Helper: aid station overlay data
# =========================================================
def _aid_station_points_for_profile(df_gpx, aid_stations: list[dict]) -> list[dict]:
    """
    Returns list of dicts: {name, km, elev_m} for stations that fall within course distance.
    Uses interpolation on smoothed elevation if present, else raw.
    """
    if not aid_stations:
        return []

    if "cum_distance" not in df_gpx.columns:
        return []

    x_km = df_gpx["cum_distance"].to_numpy(dtype=float) / 1000.0
    if len(x_km) < 2:
        return []

    y = None
    if "elev_smooth" in df_gpx.columns:
        y = df_gpx["elev_smooth"].to_numpy(dtype=float)
    elif "elev_raw" in df_gpx.columns:
        y = df_gpx["elev_raw"].to_numpy(dtype=float)
    else:
        return []

    total_km = float(x_km[-1])
    pts = []
    for a in aid_stations:
        try:
            km = float(a.get("km"))
            name = str(a.get("name", "Aid station"))
        except Exception:
            continue

        if km < 0 or km > total_km:
            continue

        elev_m = float(np.interp(km, x_km, y))
        pts.append({"name": name, "km": km, "elev_m": elev_m})

    pts.sort(key=lambda d: d["km"])
    return pts


# =========================================================
# Tabs
# =========================================================
tab_course, tab_segments, tab_strategy, tab_fueling = st.tabs(
    ["📍 Course", "⛰️ Segments & Climbs", "🧠 Strategy", "🍌 Fueling & Mental"]
)

with tab_course:
    st.subheader("Course overview")
    overview = result.overview_df.copy()
    overview.columns = [c.replace("_", " ").title() for c in overview.columns]
    st.dataframe(overview, use_container_width=True, hide_index=True)

    st.subheader("Elevation profile")

    df = result.df_gpx.copy()
    if "cum_distance" not in df.columns:
        st.warning("cum_distance missing in df_gpx")
        st.stop()

    x_km = df["cum_distance"].to_numpy(dtype=float) / 1000.0
    y = df["elev_smooth"].to_numpy(dtype=float) if "elev_smooth" in df.columns else df["elev_raw"].to_numpy(dtype=float)

    # Aid station toggle only when race library + aid stations exist
    aid_stations = []
    if "Race library:" in (st.session_state.get("last_source_label") or ""):
        aid_stations = st.session_state.get("picked_race_aid_stations") or []

    aid_pts = _aid_station_points_for_profile(df, aid_stations)
    show_aids = False
    if aid_pts:
        show_aids = st.toggle("Show aid stations", value=True)

    # Prefer Plotly if installed; fallback to matplotlib/line_chart
    try:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()

        # elevation line
        fig.add_trace(
            go.Scatter(
                x=x_km,
                y=y,
                mode="lines",
                name="Elevation",
                hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y:.0f} m<extra></extra>",
            )
        )

        # optional debug raw overlay
        if show_debug and "elev_raw" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_km,
                    y=df["elev_raw"].to_numpy(dtype=float),
                    mode="lines",
                    name="Raw",
                    opacity=0.35,
                    hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y:.0f} m<extra></extra>",
                )
            )

        # aid stations overlay
        if show_aids and aid_pts:
            xs = [p["km"] for p in aid_pts]
            ys = [p["elev_m"] for p in aid_pts]
            labels = [p["name"] for p in aid_pts]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    name="Aid stations",
                    text=labels,
                    textposition="top center",
                    hovertemplate="%{text}<br>km %{x:.1f}<br>Elevation %{y:.0f} m<extra></extra>",
                )
            )

        fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            hovermode="x unified",
        )
        fig.update_xaxes(rangeslider=dict(visible=True))
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        # fallback
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(x_km, y)

        if show_aids and aid_pts:
            ax.scatter([p["km"] for p in aid_pts], [p["elev_m"] for p in aid_pts])
            # keep labels minimal in matplotlib
            for p in aid_pts:
                ax.annotate(p["name"], (p["km"], p["elev_m"]), fontsize=8)

        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Elevation (m)")
        st.pyplot(fig, clear_figure=True)

    # aid station table
    if show_aids and aid_pts:
        st.markdown("### Aid stations")
        import pandas as pd

        aid_df = pd.DataFrame(aid_pts)
        aid_df.rename(columns={"km": "km (distance)", "elev_m": "elevation (m)"}, inplace=True)
        st.dataframe(aid_df, use_container_width=True, hide_index=True)

with tab_segments:
    st.subheader("Segments & climbs")
    st.dataframe(result.segments_df, use_container_width=True, hide_index=True)

    st.markdown("### Key climbs")
    st.dataframe(result.climbs_df, use_container_width=True, hide_index=True)

with tab_strategy:
    st.subheader("Race strategy")
    if result.strategy_text:
        st.markdown(result.strategy_text)
    else:
        st.info("No strategy text available.")

    if result.strategy_tables:
        if "critical_sections" in result.strategy_tables:
            st.markdown("### Critical sections")
            st.dataframe(result.strategy_tables["critical_sections"], hide_index=True)

        if "pacing_chunks" in result.strategy_tables:
            st.markdown("### Pacing chunks")
            st.dataframe(result.strategy_tables["pacing_chunks"], hide_index=True)

with tab_fueling:
    st.subheader("Fueling & mental cues")
    data = result.strategy_data or {}

    fp = data.get("fueling_plan")
    if fp:
        st.metric("Carbs (g/hr)", fp.get("carbs_g_per_hour", "—"))
        st.write(fp.get("hydration_notes", "—"))

    cues = data.get("mental_cues", [])
    if cues:
        st.markdown("### Mental cues")
        for c in cues:
            st.markdown(f"- **km {c.get('km')}**: {c.get('cue')}")