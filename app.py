# app.py
from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.pipeline import PipelineResult
from src.athlete_profile import get_default_athlete_profile
from src.race_strategy_generator import generate_race_strategy
from src.outputs.output_formatter import make_strategy_tables


# -------------------------
# Helpers
# -------------------------
def _file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def cached_full_run(
    gpx_bytes: bytes,
    filename: str,
    elevation_floor_m: float,
    savgol_window_length: int,
    savgol_polyorder: int,
    llm_model: str,
    llm_max_output_tokens: int,
):
    """
    Runs the full pipeline, but caches the result.
    Same file + same settings => instant return.

    NOTE: this caches the LLM call too (good for cost + speed),
    but means identical inputs will reuse the same output.
    """
    from src.pipeline import run_pipeline, PipelineConfig  # local import to keep cache stable

    with tempfile.TemporaryDirectory() as tmpdir:
        gpx_path = Path(tmpdir) / filename
        gpx_path.write_bytes(gpx_bytes)

        cfg = PipelineConfig(
            gpx_path=str(gpx_path),
            elevation_floor_m=float(elevation_floor_m),
            savgol_window_length=int(savgol_window_length),
            savgol_polyorder=int(savgol_polyorder),
            llm_model=str(llm_model),
            llm_max_output_tokens=int(llm_max_output_tokens),
        )

        return run_pipeline(cfg)


def _split_readable_and_json(text: str) -> Tuple[str, str]:
    """
    Split combined model output into:
      - readable narrative (coach notes)
      - JSON tail (optional)

    Handles:
      - ```json ...``` fenced output
      - narrative + JSON appended
      - PART 1 / PART 2 headings
    """
    if not text:
        return "", ""

    t = text.strip()

    # 1) Prefer fenced JSON if present
    m = re.search(r"```json\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        json_part = m.group(1).strip()
        readable = (t[: m.start()] + t[m.end() :]).strip()
    else:
        # 2) Otherwise split at first JSON object start
        # Look for a "{" that likely begins the JSON object.
        idx = t.find("{")
        if idx != -1:
            readable = t[:idx].strip()
            json_part = t[idx:].strip()
        else:
            readable, json_part = t, ""

    # Remove possible PART headings
    readable = re.sub(r"(?im)^\s*PART\s*1\s*—.*$", "", readable).strip()
    readable = re.sub(r"(?im)^\s*PART\s*2\s*—.*$", "", readable).strip()

    # Belt + suspenders: if readable still looks like JSON, blank it
    if readable.startswith("{") or readable.startswith("[") or "{\n" in readable:
        readable = ""

    return readable, json_part


def _init_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_file_name", None)
    st.session_state.setdefault("last_file_digest", None)


_init_state()


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Race Strategy Generator", layout="wide")

st.title("🏃‍♂️ Race Strategy Generator")
st.caption("Upload a GPX file and generate a personalized race strategy.")


# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Inputs")
gpx_file = st.sidebar.file_uploader("Upload GPX file", type=["gpx"])

st.sidebar.header("GPX cleaning")
elevation_floor = st.sidebar.number_input(
    "Elevation floor (m)", min_value=0, max_value=2000, value=200, step=10
)
savgol_window = st.sidebar.number_input(
    "Savgol window length (odd)", min_value=5, max_value=101, value=13, step=2
)
savgol_polyorder = st.sidebar.number_input(
    "Savgol polyorder", min_value=1, max_value=6, value=3, step=1
)

st.sidebar.header("LLM")
llm_model = st.sidebar.text_input("Model", value="gpt-4.1-mini")
llm_max_output_tokens = st.sidebar.slider(
    "Max output tokens",
    min_value=800,
    max_value=6000,
    value=3000,
    step=100,
    help="If JSON parsing fails, increase this first.",
)

run_btn = st.sidebar.button("Run analysis", type="primary", disabled=(gpx_file is None))

st.sidebar.divider()

retry_btn = st.sidebar.button(
    "Retry strategy only (LLM)",
    help="Re-run ONLY the LLM using the already-built course summaries from the last run.",
    disabled=(st.session_state["last_result"] is None),
)

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.session_state["last_result"] = None
    st.session_state["last_file_name"] = None
    st.session_state["last_file_digest"] = None
    st.rerun()


# -------------------------
# Actions
# -------------------------
def _run_full_pipeline(upload) -> Tuple[Optional[PipelineResult], Optional[str]]:
    file_bytes = upload.getvalue()
    digest = _file_digest(file_bytes)

    try:
        with st.spinner("Running race analysis..."):
            res = cached_full_run(
                gpx_bytes=file_bytes,
                filename=upload.name,
                elevation_floor_m=float(elevation_floor),
                savgol_window_length=int(savgol_window),
                savgol_polyorder=int(savgol_polyorder),
                llm_model=str(llm_model),
                llm_max_output_tokens=int(llm_max_output_tokens),
            )
    except Exception as e:
        return None, str(e)

    st.session_state["last_result"] = res
    st.session_state["last_file_name"] = upload.name
    st.session_state["last_file_digest"] = digest
    return res, None


def _retry_llm_only() -> Tuple[Optional[PipelineResult], Optional[str]]:
    res: PipelineResult = st.session_state["last_result"]
    if res is None:
        return None, "No previous run to retry from."

    athlete_profile = get_default_athlete_profile()

    try:
        with st.spinner("Retrying strategy generation (LLM only)..."):
            strategy_text, strategy_data = generate_race_strategy(
                course_summary=res.course_summary,
                segment_summaries=res.segment_summaries,
                climb_summaries=res.climb_summaries,
                athlete_profile=athlete_profile,
                model=str(llm_model),
                max_output_tokens=int(llm_max_output_tokens),
            )
    except Exception as e:
        return None, str(e)

    res.strategy_text = strategy_text
    res.strategy_data = strategy_data

    if strategy_data is not None:
        res.strategy_tables = make_strategy_tables(strategy_data)
    else:
        res.strategy_tables = {}

    st.session_state["last_result"] = res
    return res, None


error_msg = None

if gpx_file is None and st.session_state["last_result"] is None:
    st.info("Upload a GPX file in the sidebar, configure settings, then click **Run analysis**.")
    st.stop()

if run_btn and gpx_file is not None:
    _, error_msg = _run_full_pipeline(gpx_file)

if retry_btn:
    _, error_msg = _retry_llm_only()

if error_msg:
    st.error(f"Pipeline failed: {error_msg}")

result: Optional[PipelineResult] = st.session_state["last_result"]

if result is None:
    st.info("Configure settings in the sidebar, then click **Run analysis**.")
    st.stop()


# -------------------------
# Header summary
# -------------------------
file_label = st.session_state.get("last_file_name") or "—"
file_digest = st.session_state.get("last_file_digest") or "—"

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", file_label)
    c2.metric("Digest", file_digest)
    c3.metric("Distance (km)", f"{result.course_summary.get('total_distance_km', '—')}")
    c4.metric("Elevation gain (m)", f"{result.course_summary.get('total_gain_m', '—')}")


# -------------------------
# Tabs
# -------------------------
tab_course, tab_segments, tab_strategy, tab_fueling = st.tabs(
    ["📍 Course", "⛰️ Segments & Climbs", "🧠 Strategy", "🍌 Fueling & Mental"]
)

with tab_course:
    st.subheader("Course overview")
    st.dataframe(result.overview_df, use_container_width=True)

    st.subheader("Elevation profile")
    df = result.df_gpx.copy()

    if "cum_distance" in df.columns:
        x_km = df["cum_distance"].to_numpy(dtype=float) / 1000.0
    else:
        st.warning("No cum_distance found in df_gpx.")
        st.stop()

    if "elev_smooth" in df.columns:
        elev = df["elev_smooth"].to_numpy(dtype=float)
    elif "elev_clean" in df.columns:
        elev = df["elev_clean"].to_numpy(dtype=float)
    else:
        elev = df["elev_raw"].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    ax.plot(x_km, elev)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Gradient (approx)")
    dx_m = np.gradient(df["cum_distance"].to_numpy(dtype=float))
    de_m = np.gradient(elev)

    grad_pct = (de_m / dx_m) * 100.0
    grad_pct = np.clip(grad_pct, -40, 40)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_km, grad_pct)
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Gradient (%)")
    st.pyplot(fig2, clear_figure=True)

    st.subheader("Downloads")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "Download overview.csv",
            result.overview_df.to_csv(index=False).encode("utf-8"),
            file_name="overview.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "Download segments.csv",
            result.segments_df.to_csv(index=False).encode("utf-8"),
            file_name="segments.csv",
            mime="text/csv",
        )
    with col_c:
        st.download_button(
            "Download climbs.csv",
            result.climbs_df.to_csv(index=False).encode("utf-8"),
            file_name="climbs.csv",
            mime="text/csv",
        )

with tab_segments:
    st.subheader("Segments & Climbs")
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)

    seg_df = result.segments_df.copy()

    seg_type = col1.multiselect(
        "Segment types",
        options=sorted(seg_df["type"].unique()) if "type" in seg_df.columns else [],
        default=sorted(seg_df["type"].unique()) if "type" in seg_df.columns else [],
    )

    min_abs_grad = col2.slider("Min |gradient| (%)", 0.0, 30.0, 0.0, 0.5)
    min_dist = col3.slider("Min distance (km)", 0.0, 5.0, 0.0, 0.1)

    if "type" in seg_df.columns and seg_type:
        seg_df = seg_df[seg_df["type"].isin(seg_type)]

    if "avg_gradient_pct" in seg_df.columns:
        seg_df = seg_df[seg_df["avg_gradient_pct"].abs() >= float(min_abs_grad)]

    if "distance_km" in seg_df.columns:
        seg_df = seg_df[seg_df["distance_km"] >= float(min_dist)]

    st.markdown("### Segments (filtered)")
    st.dataframe(seg_df, use_container_width=True)

    st.markdown("### Key climbs")
    st.dataframe(result.climbs_df, use_container_width=True)

with tab_strategy:
    st.subheader("Race strategy")
    data = result.strategy_data or {}

    def _safe_text(x) -> str:
        return (x or "").strip()

    def _show_card(title: str, body: str):
        st.markdown(f"**{title}**")
        st.write(body if body else "—")

    raw_text = (result.strategy_text or "").strip()
    readable, _json_tail = _split_readable_and_json(raw_text)
    readable = _safe_text(readable)

    st.markdown("### Coach notes")
    if readable:
        st.markdown(readable)
    else:
        st.info(
            "This run didn't include a separate narrative strategy. "
            "Showing the structured plan below instead."
        )

    st.divider()

    st.markdown("### Plan by race phase")
    gs = (data.get("global_strategy") or {}) if isinstance(data, dict) else {}
    if gs:
        c1, c2, c3 = st.columns(3)
        with c1:
            _show_card("🟢 Early", _safe_text(gs.get("early")))
        with c2:
            _show_card("🟡 Mid", _safe_text(gs.get("mid")))
        with c3:
            _show_card("🔴 Late", _safe_text(gs.get("late")))
    else:
        st.warning(
            "Structured strategy (global_strategy) missing. "
            "Try **Retry strategy only (LLM)** with higher token limit."
        )

    st.divider()

    st.markdown("### Critical sections")
    if result.strategy_tables and "critical_sections" in result.strategy_tables:
        st.dataframe(
            result.strategy_tables["critical_sections"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No critical sections table available for this run.")

    st.markdown("### Pacing chunks")
    if result.strategy_tables and "pacing_chunks" in result.strategy_tables:
        st.dataframe(
            result.strategy_tables["pacing_chunks"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No pacing chunks table available for this run.")

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Fueling target")
        fp = (data.get("fueling_plan") or {}) if isinstance(data, dict) else {}
        carbs = fp.get("carbs_g_per_hour", "—")
        st.metric("Carbs (g/hr)", carbs)

    with colB:
        st.markdown("### Mental cue (preview)")
        cues = (data.get("mental_cues") or []) if isinstance(data, dict) else []
        if cues:
            first = cues[0]
            st.write(f"**km {first.get('km','?')}** — {first.get('cue','')}")
        else:
            st.write("—")

    with st.expander("Debug: show parsed JSON"):
        st.json(data)

with tab_fueling:
    st.subheader("Fueling & mental cues")
    data = result.strategy_data or {}

    fp = data.get("fueling_plan") if isinstance(data, dict) else None
    cues = (data.get("mental_cues") or []) if isinstance(data, dict) else []

    st.markdown("### Fueling plan")
    if not fp:
        st.info("No fueling_plan found (likely JSON parse failed).")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Carbs (g/hr)", fp.get("carbs_g_per_hour", "—"))
        with col2:
            st.markdown("**Hydration notes**")
            st.write(fp.get("hydration_notes", "—"))

        special = fp.get("special_sections", []) or []
        if special:
            st.markdown("### Special sections")
            for item in special:
                st.markdown(
                    f"**{item.get('km_range','?')}** — {item.get('reason','')}\n\n"
                    f"- Focus: {item.get('fueling_focus','')}"
                )

    st.divider()

    st.markdown("### Mental cues")
    if not cues:
        st.info("No mental_cues found (likely JSON parse failed).")
    else:
        for c in cues:
            st.markdown(f"- **km {c.get('km','?')}**: {c.get('cue','')}")