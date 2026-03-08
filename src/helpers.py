import pandas as pd
import streamlit as st

from src.config import PLOT_CONFIG, logger


def safe_team_color(raw_color, fallback="#60A5FA"):
    if pd.notna(raw_color):
        raw = str(raw_color).replace("#", "").strip()
        if len(raw) == 6:
            return f"#{raw}"
    return fallback

def debug_print(msg):
    logger.info("[DEBUG] %s", msg)

def show_chart(fig):
    if fig is None:
        return False
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    return True

def show_chart_or_info(fig, message):
    if not show_chart(fig):
        st.info(message)

def initialize_session_state():
    st.session_state.setdefault("selected_session", None)
    st.session_state.setdefault("run_dashboard", False)

def build_driver_selection(best_laps):
    comp_source = best_laps[["driver_number"]].copy()
    comp_source["driver_label"] = (
        best_laps["name_acronym"].fillna(best_laps["full_name"]).astype(str)
        + " - "
        + best_laps["full_name"].fillna("N/A").astype(str)
        if "name_acronym" in best_laps.columns
        else best_laps["full_name"].fillna("N/A").astype(str)
    )
    comp_source = comp_source.drop_duplicates(subset=["driver_number"])
    driver_options = comp_source["driver_number"].tolist()
    driver_label_map = dict(zip(comp_source["driver_number"], comp_source["driver_label"]))
    default_drivers = driver_options[:3]
    return comp_source, driver_options, driver_label_map, default_drivers
