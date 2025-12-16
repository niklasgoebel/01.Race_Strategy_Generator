from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

from src.athlete_profile import (
    get_default_athlete_profile,
    list_athlete_profiles,
    load_athlete_profile,
    save_athlete_profile,
    delete_athlete_profile,
    get_profile_templates,
    infer_secondary_metrics,
    ensure_default_profiles_exist,
)
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
    llm_model: str,
    llm_max_output_tokens: int,
    athlete_profile_name: Optional[str] = None,
):
    from src.pipeline import PipelineConfig, run_pipeline

    # Load athlete profile
    if athlete_profile_name:
        try:
            athlete_profile = load_athlete_profile(athlete_profile_name)
        except Exception:
            athlete_profile = get_default_athlete_profile()
    else:
        athlete_profile = get_default_athlete_profile()

    with tempfile.TemporaryDirectory() as tmpdir:
        gpx_path = Path(tmpdir) / filename
        gpx_path.write_bytes(gpx_bytes)

        cfg = PipelineConfig(
            gpx_path=str(gpx_path),
            llm_model=llm_model,
            llm_max_output_tokens=llm_max_output_tokens,
            athlete_profile=athlete_profile,
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
    
    # Profile management
    st.session_state.setdefault("selected_profile_name", None)
    st.session_state.setdefault("show_profile_editor", False)
    st.session_state.setdefault("editing_profile_name", None)


_init_state()

# Initialize default profiles on first run
ensure_default_profiles_exist()


# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Race Strategy Generator", layout="wide")
st.title("🏃‍♂️ Race Strategy Generator")
st.caption("Select a race or upload a GPX file to generate a personalized strategy.")


# =========================================================
# Sidebar — athlete profile selection
# =========================================================
st.sidebar.header("👤 Athlete Profile")

# Get available profiles
available_profiles = list_athlete_profiles()

if not available_profiles:
    st.sidebar.warning("No profiles found. Create one below.")
    selected_profile_name = None
else:
    # Profile selector
    if st.session_state["selected_profile_name"] not in available_profiles:
        st.session_state["selected_profile_name"] = available_profiles[0]
    
    selected_profile_name = st.sidebar.selectbox(
        "Select profile",
        available_profiles,
        index=available_profiles.index(st.session_state["selected_profile_name"]) 
            if st.session_state["selected_profile_name"] in available_profiles else 0,
        key="profile_selector"
    )
    st.session_state["selected_profile_name"] = selected_profile_name

# Profile management buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("➕ New", help="Create new profile"):
        st.session_state["show_profile_editor"] = True
        st.session_state["editing_profile_name"] = None
with col2:
    if st.button("✏️ Edit", help="Edit current profile", disabled=not selected_profile_name):
        st.session_state["show_profile_editor"] = True
        st.session_state["editing_profile_name"] = selected_profile_name
with col3:
    if st.button("🗑️ Delete", help="Delete current profile", disabled=not selected_profile_name):
        if selected_profile_name:
            if delete_athlete_profile(selected_profile_name):
                st.sidebar.success(f"Deleted '{selected_profile_name}'")
                st.session_state["selected_profile_name"] = None
                st.rerun()
            else:
                st.sidebar.error("Failed to delete profile")

# Profile editor modal (expander)
if st.session_state.get("show_profile_editor", False):
    with st.sidebar.expander("Profile Editor", expanded=True):
        editing_name = st.session_state.get("editing_profile_name")
        
        # Load existing profile or start with template
        if editing_name:
            st.caption(f"Editing: {editing_name}")
            profile_data = load_athlete_profile(editing_name)
        else:
            st.caption("Create new profile")
            # Offer templates
            templates = get_profile_templates()
            template_names = ["Start from scratch"] + list(templates.keys())
            selected_template = st.selectbox("Start from template:", template_names)
            
            if selected_template != "Start from scratch":
                profile_data = templates[selected_template].copy()
            else:
                profile_data = {
                    "name": "",
                    "experience": "",
                    "weekly_volume_km": 60,
                    "long_run_km": 20,
                    "vo2max": 50,
                    "max_hr": 190,
                    "lactate_threshold_hr": 170,
                    "lactate_threshold_pace_per_km": "4:30",
                    "goal_type": "finish strong",
                    "fuel_type": "gels + sports drink",
                    "carbs_per_hour_target_g": 60,
                }
        
        # Dropdown options
        EXPERIENCE_OPTIONS = [
            "Beginner/First Ultra",
            "Intermediate (some ultras)",
            "Experienced (multiple ultras)",
            "Advanced/Elite"
        ]
        
        GOAL_TYPE_OPTIONS = [
            "Finish comfortably",
            "Smart pacing, finish strong",
            "Strong performance",
            "Competitive/PR attempt"
        ]
        
        FUEL_TYPE_OPTIONS = [
            "Gels + sports drink",
            "Mixed (gels, bars, real food)",
            "Primarily real food"
        ]
        
        # Advanced mode toggle (MUST be outside form for immediate effect)
        show_advanced = st.checkbox("Show advanced fields", value=st.session_state.get("show_advanced_fields", False), key="show_advanced_toggle")
        st.session_state["show_advanced_fields"] = show_advanced
        
        # Profile form
        with st.form("profile_form"):
            new_name = st.text_input("Profile Name", value=profile_data.get("name", ""))
            
            st.markdown("**Basic Info**")
            col_a, col_b = st.columns(2)
            with col_a:
                # Map existing experience to dropdown options
                current_exp = profile_data.get("experience", "")
                exp_index = 0  # default to first option
                if "beginner" in current_exp.lower() or "first" in current_exp.lower() or "newer" in current_exp.lower():
                    exp_index = 0
                elif "advanced" in current_exp.lower() or "elite" in current_exp.lower():
                    exp_index = 3
                elif "experienced" in current_exp.lower() or "multiple" in current_exp.lower():
                    exp_index = 2
                elif "intermediate" in current_exp.lower():
                    exp_index = 1
                
                experience = st.selectbox("Experience level", EXPERIENCE_OPTIONS, index=exp_index)
                weekly_vol = st.number_input("Weekly volume (km)", value=int(profile_data.get("weekly_volume_km", 60)), min_value=0)
            with col_b:
                long_run = st.number_input("Long run (km)", value=int(profile_data.get("long_run_km", 20)), min_value=0)
            
            st.markdown("**Goals & Preferences**")
            # Map existing goal type to dropdown options
            current_goal = profile_data.get("goal_type", "")
            goal_index = 0  # default to first option
            if "finish" in current_goal.lower() and "comfortably" in current_goal.lower():
                goal_index = 0
            elif "smart" in current_goal.lower() or "strong" in current_goal.lower():
                goal_index = 1
            elif "performance" in current_goal.lower():
                goal_index = 2
            elif "competitive" in current_goal.lower() or "pr" in current_goal.lower() or "race" in current_goal.lower():
                goal_index = 3
            
            goal_type = st.selectbox("Goal type", GOAL_TYPE_OPTIONS, index=goal_index)
            
            st.markdown("**Fueling**")
            # Map existing fuel type to dropdown options
            current_fuel = profile_data.get("fuel_type", "")
            fuel_index = 0  # default to first option
            if "mixed" in current_fuel.lower() or "bars" in current_fuel.lower():
                fuel_index = 1
            elif "real food" in current_fuel.lower():
                fuel_index = 2
            
            fuel_type = st.selectbox("Fuel type", FUEL_TYPE_OPTIONS, index=fuel_index)
            
            # Advanced fields (conditionally shown)
            if show_advanced:
                st.markdown("**Advanced Metrics**")
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    vo2max = st.number_input("VO2max (optional)", value=int(profile_data.get("vo2max", 0)) if profile_data.get("vo2max") else 0, min_value=0, max_value=90, help="Leave at 0 to auto-calculate")
                    max_hr = st.number_input("Max HR (optional)", value=int(profile_data.get("max_hr", 0)) if profile_data.get("max_hr") else 0, min_value=0, max_value=220, help="Leave at 0 to auto-calculate")
                with col_adv2:
                    lt_hr = st.number_input("Lactate Threshold HR (optional)", value=int(profile_data.get("lactate_threshold_hr", 0)) if profile_data.get("lactate_threshold_hr") else 0, min_value=0, max_value=220, help="Leave at 0 to auto-calculate")
                    lt_pace = st.text_input("LT Pace (mm:ss per km, optional)", value=str(profile_data.get("lactate_threshold_pace_per_km", "")), help="Leave empty to auto-calculate")
            else:
                # Hidden but need to preserve values if they exist
                vo2max = int(profile_data.get("vo2max", 0)) if profile_data.get("vo2max") else 0
                max_hr = int(profile_data.get("max_hr", 0)) if profile_data.get("max_hr") else 0
                lt_hr = int(profile_data.get("lactate_threshold_hr", 0)) if profile_data.get("lactate_threshold_hr") else 0
                lt_pace = str(profile_data.get("lactate_threshold_pace_per_km", ""))
            
            col_save, col_cancel = st.columns(2)
            with col_save:
                submitted = st.form_submit_button("💾 Save", type="primary")
            with col_cancel:
                cancelled = st.form_submit_button("Cancel")
            
            if submitted:
                if not new_name:
                    st.error("Profile name is required")
                else:
                    # Build profile dict with required fields
                    new_profile = {
                        "name": new_name,
                        "target_race": profile_data.get("target_race", ""),
                        "race_date": profile_data.get("race_date", ""),
                        "experience": experience,
                        "weekly_volume_km": weekly_vol,
                        "long_run_km": long_run,
                        "recent_best_5k": profile_data.get("recent_best_5k", ""),
                        "recent_best_half_marathon": profile_data.get("recent_best_half_marathon", ""),
                        "preferred_ascent_effort": profile_data.get("preferred_ascent_effort", "steady and controlled"),
                        "descent_style": profile_data.get("descent_style", "confident"),
                        "heat_tolerance": profile_data.get("heat_tolerance", "average"),
                        "goal_type": goal_type,
                        "fuel_type": fuel_type,
                        "hydration_notes": profile_data.get("hydration_notes", "carry water, refill at aid stations"),
                    }
                    
                    # Add optional advanced fields only if provided (non-zero/non-empty)
                    if vo2max > 0:
                        new_profile["vo2max"] = vo2max
                    if max_hr > 0:
                        new_profile["max_hr"] = max_hr
                    if lt_hr > 0:
                        new_profile["lactate_threshold_hr"] = lt_hr
                    if lt_pace and lt_pace.strip():
                        new_profile["lactate_threshold_pace_per_km"] = lt_pace
                    
                    try:
                        # Use infer_secondary_metrics to fill in missing values
                        new_profile = infer_secondary_metrics(new_profile)
                        save_athlete_profile(new_profile, new_name)
                        st.success(f"Saved profile '{new_name}'!")
                        st.session_state["show_profile_editor"] = False
                        st.session_state["selected_profile_name"] = new_name
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
            
            if cancelled:
                st.session_state["show_profile_editor"] = False
                st.rerun()

st.sidebar.divider()

# =========================================================
# Sidebar — source selection
# =========================================================
st.sidebar.header("📍 Course Selection")

source_mode = st.sidebar.radio(
    "Choose data source",
    ["Race library (beta)", "Upload GPX"],
    index=0,
)

# -------------------------
# Application settings
# -------------------------
# Smoothing parameters are auto-determined from GPX data characteristics
# LLM settings use sensible defaults
llm_model = "gpt-4.1-mini"
llm_max_output_tokens = 3000


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

# Export section
if st.session_state["last_result"] is not None:
    st.sidebar.divider()
    st.sidebar.subheader("📥 Export Strategy")
    
    result = st.session_state["last_result"]
    
    # Get selected profile for exports
    if st.session_state.get("selected_profile_name"):
        try:
            export_profile = load_athlete_profile(st.session_state["selected_profile_name"])
        except Exception:
            export_profile = get_default_athlete_profile()
    else:
        export_profile = get_default_athlete_profile()
    
    # PDF Exports
    from src.outputs.export_pdf import generate_race_day_pdf, generate_cheat_sheet_pdf
    from src.outputs.export_mobile import generate_mobile_html, generate_json_export
    
    try:
        # Full PDF
        full_pdf_bytes = generate_race_day_pdf(result, export_profile)
        st.sidebar.download_button(
            label="📄 Full Strategy (PDF)",
            data=full_pdf_bytes,
            file_name=f"race_strategy_{export_profile.get('name', 'athlete')}.pdf",
            mime="application/pdf",
        )
        
        # Cheat Sheet PDF
        cheat_pdf_bytes = generate_cheat_sheet_pdf(result, export_profile)
        st.sidebar.download_button(
            label="📋 Cheat Sheet (PDF)",
            data=cheat_pdf_bytes,
            file_name=f"race_cheatsheet_{export_profile.get('name', 'athlete')}.pdf",
            mime="application/pdf",
        )
        
        # Mobile HTML
        mobile_html = generate_mobile_html(result, export_profile)
        st.sidebar.download_button(
            label="📱 Mobile View (HTML)",
            data=mobile_html,
            file_name=f"race_strategy_mobile_{export_profile.get('name', 'athlete')}.html",
            mime="text/html",
        )
        
        # JSON Export
        json_export = generate_json_export(result, export_profile)
        st.sidebar.download_button(
            label="💾 Raw Data (JSON)",
            data=json_export,
            file_name=f"race_strategy_{export_profile.get('name', 'athlete')}.json",
            mime="application/json",
        )
    except Exception as e:
        st.sidebar.error(f"Export error: {e}")

st.sidebar.divider()

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
                llm_model=llm_model,
                llm_max_output_tokens=llm_max_output_tokens,
                athlete_profile_name=st.session_state.get("selected_profile_name"),
            )
    except Exception as e:
        return None, str(e)

    st.session_state["last_result"] = res
    st.session_state["last_source_label"] = source_label
    st.session_state["last_file_name"] = filename
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
c1.metric("Source", st.session_state.get("last_source_label", "-"))
c2.metric("File", st.session_state.get("last_file_name", "-"))
c3.metric("Distance", f"{result.course_summary.get('total_distance_km', '-')} km")
c4.metric("Elevation gain", f"{result.course_summary.get('total_gain_m', '-')} m")


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
tab_course, tab_segments, tab_strategy, tab_time, tab_fueling, tab_compare = st.tabs(
    ["📍 Course", "⛰️ Segments & Climbs", "🧠 Strategy", "⏱️ Time Estimates", "🍌 Fueling & Mental", "🔄 Compare"]
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

    # Chart enhancement toggles
    col_toggle1, col_toggle2 = st.columns(2)
    with col_toggle1:
        show_strategy_overlay = st.checkbox("Show effort zones", value=True)
    with col_toggle2:
        show_strategy_markers = st.checkbox("Show strategy markers", value=True)
    
    # Prefer Plotly if installed; fallback to matplotlib/line_chart
    try:
        import plotly.graph_objects as go  # type: ignore
        from src.outputs.chart_enhancements import (
            create_strategy_overlay_data,
            generate_marker_positions,
            add_critical_sections_to_chart,
        )

        fig = go.Figure()
        
        # Strategy overlays (effort zones) - add FIRST so they're behind
        if show_strategy_overlay and result.strategy_data:
            overlays = create_strategy_overlay_data(result.segments_df, result.strategy_data)
            for overlay in overlays:
                fig.add_vrect(
                    x0=overlay["x0"],
                    x1=overlay["x1"],
                    fillcolor=overlay["color"],
                    layer="below",
                    line_width=0,
                    annotation_text=overlay["label"],
                    annotation_position="top left",
                    annotation_font_size=8,
                    annotation_font_color="rgba(255, 255, 255, 0.5)",
                )

        # elevation line
        fig.add_trace(
            go.Scatter(
                x=x_km,
                y=y,
                mode="lines",
                name="Elevation",
                line=dict(color="#667eea", width=2),
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
                    marker=dict(size=10, color="#4ade80", symbol="circle"),
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=9),
                    hovertemplate="%{text}<br>km %{x:.1f}<br>Elevation %{y:.0f} m<extra></extra>",
                )
            )
        
        # Strategy markers (fueling and mental cues)
        if show_strategy_markers and result.strategy_data:
            marker_x, marker_y, marker_labels, marker_types = generate_marker_positions(
                result.strategy_data, df
            )
            
            if marker_x:
                # Separate fueling and mental cue markers for different colors
                fueling_x = [marker_x[i] for i in range(len(marker_x)) if marker_types[i] == "fueling"]
                fueling_y = [marker_y[i] for i in range(len(marker_y)) if marker_types[i] == "fueling"]
                fueling_labels = [marker_labels[i] for i in range(len(marker_labels)) if marker_types[i] == "fueling"]
                
                if fueling_x:
                    fig.add_trace(
                        go.Scatter(
                            x=fueling_x,
                            y=fueling_y,
                            mode="markers",
                            name="Fueling",
                            marker=dict(size=12, color="#fbbf24", symbol="diamond"),
                            text=fueling_labels,
                            hovertemplate="%{text}<br>km %{x:.1f}<extra></extra>",
                        )
                    )
                
                mental_x = [marker_x[i] for i in range(len(marker_x)) if marker_types[i] == "mental_cue"]
                mental_y = [marker_y[i] for i in range(len(marker_y)) if marker_types[i] == "mental_cue"]
                mental_labels = [marker_labels[i] for i in range(len(marker_labels)) if marker_types[i] == "mental_cue"]
                
                if mental_x:
                    fig.add_trace(
                        go.Scatter(
                            x=mental_x,
                            y=mental_y,
                            mode="markers",
                            name="Mental Cues",
                            marker=dict(size=12, color="#a78bfa", symbol="star"),
                            text=mental_labels,
                            hovertemplate="%{text}<br>km %{x:.1f}<extra></extra>",
                        )
                    )

        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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

with tab_time:
    st.subheader("Time Estimates & Pacing")
    
    if result.finish_time_range:
        st.markdown("### Predicted Finish Time")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Conservative",
                result.finish_time_range["conservative_str"],
                help="Cautious pacing with buffer"
            )
        with col2:
            st.metric(
                "Expected",
                result.finish_time_range["expected_str"],
                help="Based on your profile and course analysis"
            )
        with col3:
            st.metric(
                "Aggressive",
                result.finish_time_range["aggressive_str"],
                help="Strong execution required"
            )
        
        st.info(f"💡 These estimates are based on your VO2max ({selected_profile_name or 'profile'}) and course characteristics. Actual time will vary with conditions and race-day execution.")
    else:
        st.info("Time estimates not available. Ensure a profile is selected.")
    
    # Aid station splits
    if result.aid_station_splits is not None and not result.aid_station_splits.empty:
        st.markdown("### Aid Station Splits")
        st.dataframe(
            result.aid_station_splits[["name", "km", "cumulative_time_str"]],
            hide_index=True,
            column_config={
                "name": "Aid Station",
                "km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
                "cumulative_time_str": "Estimated Time",
            }
        )
    
    # Pacing calculator
    if result.finish_time_range and result.segments_with_times is not None:
        st.markdown("### Pacing Calculator")
        st.caption("Adjust your target finish time to see if it's realistic for your profile.")
        
        # Get athlete profile for reverse calculation
        if st.session_state.get("selected_profile_name"):
            try:
                calc_profile = load_athlete_profile(st.session_state["selected_profile_name"])
            except Exception:
                calc_profile = get_default_athlete_profile()
        else:
            calc_profile = get_default_athlete_profile()
        
        # Time slider
        expected_min = result.finish_time_range["expected_min"]
        min_time = int(expected_min * 0.7)  # 30% faster
        max_time = int(expected_min * 1.5)  # 50% slower
        
        target_time_min = st.slider(
            "Target finish time (hours:minutes)",
            min_value=min_time,
            max_value=max_time,
            value=int(expected_min),
            step=5,
            format="%d min"
        )
        
        from src.time_estimator import reverse_calculate_required_pace, format_time_hhmm
        
        target_time_str = format_time_hhmm(target_time_min)
        st.write(f"**Target:** {target_time_str}")
        
        # Calculate feasibility (use raw segment data, not formatted display version)
        pace_analysis = reverse_calculate_required_pace(
            target_time_min,
            result.seg,  # Use raw segments with original column names
            calc_profile
        )
        
        if pace_analysis["feasible"]:
            speedup_pct = pace_analysis["speedup_pct"]
            
            if speedup_pct > 0:
                st.success(f"✅ {pace_analysis['feasibility']}")
                st.write(f"This is **{abs(speedup_pct):.1f}% slower** than your expected pace.")
            elif speedup_pct > -5:
                st.success(f"✅ {pace_analysis['feasibility']}")
                st.write(f"This matches your expected pacing closely.")
            elif speedup_pct > -15:
                st.warning(f"⚠️ {pace_analysis['feasibility']}")
                st.write(f"This requires **{abs(speedup_pct):.1f}% faster** pacing than expected.")
            else:
                st.error(f"❌ {pace_analysis['feasibility']}")
                st.write(f"This requires **{abs(speedup_pct):.1f}% faster** pacing - may not be achievable.")
        else:
            st.error("Unable to calculate pacing requirements.")
    
    # Segment-level time breakdown (expandable)
    if result.segments_with_times is not None:
        with st.expander("📊 Detailed Segment Times"):
            st.dataframe(
                result.segments_with_times[[
                    "type", "start_km", "end_km", "distance_km", 
                    "avg_gradient", "estimated_time_min", "pace_min_per_km"
                ]],
                hide_index=True,
                column_config={
                    "type": "Type",
                    "start_km": st.column_config.NumberColumn("Start (km)", format="%.1f"),
                    "end_km": st.column_config.NumberColumn("End (km)", format="%.1f"),
                    "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
                    "avg_gradient": st.column_config.NumberColumn("Gradient (%)", format="%.1f"),
                    "estimated_time_min": st.column_config.NumberColumn("Time (min)", format="%.1f"),
                    "pace_min_per_km": st.column_config.NumberColumn("Pace (min/km)", format="%.2f"),
                }
            )

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

with tab_compare:
    st.subheader("Strategy Comparison")
    st.caption("Compare strategies for different athlete profiles on the same course")
    
    available_profiles = list_athlete_profiles()
    
    if len(available_profiles) < 2:
        st.info("📝 Create at least 2 profiles to use comparison feature. Go to the sidebar to create more profiles.")
    else:
        # Profile selectors
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### Profile A")
            default_a = st.session_state.get("selected_profile_name", available_profiles[0])
            if default_a not in available_profiles:
                default_a = available_profiles[0]
            
            profile_a_name = st.selectbox(
                "Select Profile A",
                available_profiles,
                index=available_profiles.index(default_a),
                key="compare_profile_a"
            )
        
        with col_b:
            st.markdown("### Profile B")
            # Default to second profile
            default_b = available_profiles[1] if len(available_profiles) > 1 else available_profiles[0]
            if st.session_state.get("compare_profile_b") and st.session_state["compare_profile_b"] in available_profiles:
                default_b = st.session_state["compare_profile_b"]
            
            profile_b_name = st.selectbox(
                "Select Profile B",
                available_profiles,
                index=available_profiles.index(default_b),
                key="compare_profile_b"
            )
        
        if profile_a_name == profile_b_name:
            st.warning("⚠️ Please select two different profiles to compare.")
        else:
            # Run comparison button
            compare_btn = st.button("🔄 Run Comparison", type="primary")
            
            if compare_btn or st.session_state.get("comparison_results"):
                # Check if we need to regenerate
                need_regenerate = (
                    compare_btn or
                    not st.session_state.get("comparison_results") or
                    st.session_state.get("last_compare_a") != profile_a_name or
                    st.session_state.get("last_compare_b") != profile_b_name
                )
                
                if need_regenerate:
                    with st.spinner("Generating strategies for both profiles..."):
                        # Get current race bytes
                        if st.session_state.get("picked_race_id"):
                            race_id = st.session_state["picked_race_id"]
                            gpx_bytes = load_race_gpx_bytes(race_id)
                            filename = st.session_state.get("picked_race_gpx_name", "race.gpx")
                        else:
                            st.error("Please run an analysis first before comparing profiles.")
                            st.stop()
                        
                        # Run pipeline for both profiles
                        try:
                            result_a = cached_full_run(
                                gpx_bytes=gpx_bytes,
                                filename=filename,
                                llm_model=llm_model,
                                llm_max_output_tokens=llm_max_output_tokens,
                                athlete_profile_name=profile_a_name,
                            )
                            
                            result_b = cached_full_run(
                                gpx_bytes=gpx_bytes,
                                filename=filename,
                                llm_model=llm_model,
                                llm_max_output_tokens=llm_max_output_tokens,
                                athlete_profile_name=profile_b_name,
                            )
                            
                            from src.comparison import (
                                compare_strategies,
                                create_comparison_dataframe,
                                highlight_differences,
                            )
                            
                            comparison = compare_strategies(result_a, result_b)
                            comparison_df = create_comparison_dataframe(
                                result_a, result_b, profile_a_name, profile_b_name
                            )
                            differences = highlight_differences(
                                result_a.strategy_data or {},
                                result_b.strategy_data or {}
                            )
                            
                            st.session_state["comparison_results"] = {
                                "result_a": result_a,
                                "result_b": result_b,
                                "comparison": comparison,
                                "comparison_df": comparison_df,
                                "differences": differences,
                            }
                            st.session_state["last_compare_a"] = profile_a_name
                            st.session_state["last_compare_b"] = profile_b_name
                        
                        except Exception as e:
                            st.error(f"Comparison failed: {e}")
                            st.stop()
                
                # Display comparison results
                comp_data = st.session_state["comparison_results"]
                
                # Key metrics comparison
                st.markdown("### 📊 Key Metrics Comparison")
                st.dataframe(comp_data["comparison_df"], use_container_width=True, hide_index=True)
                
                # Time difference highlight
                if comp_data["comparison"]["time_differences"]["available"]:
                    time_diff = comp_data["comparison"]["time_differences"]
                    diff_min = time_diff["difference_min"]
                    
                    st.markdown("### ⏱️ Finish Time Difference")
                    
                    if abs(diff_min) < 5:
                        st.info(f"✨ Very similar finish times (within {abs(diff_min):.1f} minutes)")
                    else:
                        faster = time_diff["faster"]
                        st.success(f"🏃 **{faster}** is estimated **{abs(diff_min):.0f} minutes** faster ({abs(time_diff['difference_pct']):.1f}%)")
                
                # Critical sections comparison
                if comp_data["comparison"]["critical_sections_diff"]:
                    st.markdown("### ⛰️ Critical Sections - Effort Level Comparison")
                    
                    sections_comp = comp_data["comparison"]["critical_sections_diff"]
                    
                    for section in sections_comp:
                        if section["different"]:
                            with st.expander(f"🔍 {section['label']} (km {section['km_range']})", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**{profile_a_name}**")
                                    st.write(f"Effort: RPE {section['effort1']}")
                                    st.write(f"HR: {section['hr1']}")
                                    if section['notes1']:
                                        st.caption(section['notes1'])
                                with col2:
                                    st.markdown(f"**{profile_b_name}**")
                                    st.write(f"Effort: RPE {section['effort2']}")
                                    st.write(f"HR: {section['hr2']}")
                                    if section['notes2']:
                                        st.caption(section['notes2'])
                
                # Strategy differences summary
                st.markdown("### 🎯 Key Strategy Differences")
                
                diffs = comp_data["differences"]
                has_differences = any(diffs.values())
                
                if not has_differences:
                    st.info("Strategies are very similar for both profiles.")
                else:
                    if diffs["pacing"]:
                        st.markdown("**Pacing:**")
                        for diff in diffs["pacing"]:
                            st.write(f"- {diff}")
                    
                    if diffs["fueling"]:
                        st.markdown("**Fueling:**")
                        for diff in diffs["fueling"]:
                            st.write(f"- {diff}")
                    
                    if diffs["mental_approach"]:
                        st.markdown("**Mental Approach:**")
                        for diff in diffs["mental_approach"]:
                            st.write(f"- {diff}")