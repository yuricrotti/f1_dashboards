import requests
import pandas as pd
import numpy as np
import streamlit as st

from src.config import APP_CONFIG, BASE
from src.helpers import debug_print, safe_team_color


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_short)
def get_json(endpoint, params=None):
    url = f"{BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_long)
def get_sessions(year):
    sessions = pd.DataFrame(get_json("sessions", {"year": year}))
    if sessions.empty:
        return sessions

    if "date_start" in sessions.columns:
        sessions["date_start"] = pd.to_datetime(sessions["date_start"], errors="coerce", utc=True)
        sessions = sessions.sort_values("date_start")

    if "session_key" in sessions.columns:
        sessions = sessions.drop_duplicates(subset=["session_key"])

    return sessions.reset_index(drop=True)

def filter_sessions_by_country(sessions, country_filter):
    if sessions.empty or not country_filter:
        return sessions
    if "country_name" not in sessions.columns:
        return sessions
    mask = sessions["country_name"].fillna("").str.contains(country_filter.strip(), case=False, regex=False)
    return sessions.loc[mask].reset_index(drop=True)

def format_session_option(session):
    date_start = pd.to_datetime(session.get("date_start"), errors="coerce", utc=True)
    date_label = date_start.strftime("%Y-%m-%d") if pd.notna(date_start) else "sem data"
    country = session.get("country_name", "N/A")
    meeting = session.get("meeting_name", "N/A")
    session_name = session.get("session_name", "N/A")
    session_key = session.get("session_key", "N/A")
    return f"{date_label} | {country} | {meeting} | {session_name} | key={session_key}"

def normalize_session_type(session):
    session_type = (session.get("session_type") or session.get("session_name") or "").strip().lower()

    if "race" in session_type:
        return "race"
    if "qualifying" in session_type or "quali" in session_type:
        return "qualifying"
    if "practice" in session_type or "fp" in session_type:
        return "practice"

    return session_type

@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_short)
def fetch_session_data(session_key, session_type):
    drivers = pd.DataFrame(get_json("drivers", {"session_key": session_key}))
    laps = pd.DataFrame(get_json("laps", {"session_key": session_key}))
    location = pd.DataFrame(get_json("location", {"session_key": session_key}))
    position = pd.DataFrame(get_json("position", {"session_key": session_key}))
    stints = pd.DataFrame(get_json("stints", {"session_key": session_key}))
    weather = pd.DataFrame(get_json("weather", {"session_key": session_key}))
    race_control = pd.DataFrame(get_json("race_control", {"session_key": session_key}))
    session_result = pd.DataFrame(get_json("session_result", {"session_key": session_key}))

    intervals = pd.DataFrame()
    if session_type == "race":
        intervals = pd.DataFrame(get_json("intervals", {"session_key": session_key}))

    debug_print(
        f"session_key={session_key} | drivers={drivers.shape} laps={laps.shape} location={location.shape} position={position.shape}"
    )

    return {
        "drivers": drivers,
        "laps": laps,
        "location": location,
        "position": position,
        "stints": stints,
        "weather": weather,
        "race_control": race_control,
        "session_result": session_result,
        "intervals": intervals,
    }

def get_race_winner(session_result, drivers):
    if session_result.empty:
        return None

    pos_col = next((c for c in ["position", "classification_position", "rank"] if c in session_result.columns), None)
    if pos_col is None:
        return None

    df = session_result.copy()
    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")
    df = df.dropna(subset=[pos_col]).sort_values(pos_col)
    if df.empty:
        return None

    winner = df.iloc[0].copy()
    driver_number = winner.get("driver_number", None)
    full_name = winner.get("full_name", None)
    team_name = winner.get("team_name", None)

    if (pd.isna(full_name) or not str(full_name).strip()) and driver_number is not None and not drivers.empty:
        dcols = [c for c in ["driver_number", "full_name", "team_name"] if c in drivers.columns]
        if dcols:
            dmeta = drivers[dcols].drop_duplicates(subset=["driver_number"]).copy()
            dmeta["driver_number"] = pd.to_numeric(dmeta["driver_number"], errors="coerce")
            row = dmeta[dmeta["driver_number"] == pd.to_numeric(driver_number, errors="coerce")]
            if not row.empty:
                if pd.isna(full_name) or not str(full_name).strip():
                    full_name = row.iloc[0].get("full_name", full_name)
                if pd.isna(team_name) or not str(team_name).strip():
                    team_name = row.iloc[0].get("team_name", team_name)

    return {
        "position": int(winner[pos_col]),
        "full_name": str(full_name) if pd.notna(full_name) else f"Driver {driver_number}",
        "team_name": str(team_name) if pd.notna(team_name) else "N/A",
        "driver_number": driver_number,
    }

def prepare_lap_times(laps, drivers):
    if laps.empty:
        return pd.DataFrame()

    needed = ["driver_number", "lap_number", "lap_duration"]
    if any(col not in laps.columns for col in needed):
        return pd.DataFrame()

    lap_times = laps.dropna(subset=needed).copy()
    if lap_times.empty:
        return pd.DataFrame()

    driver_cols = [
        col
        for col in ["driver_number", "full_name", "name_acronym", "team_name", "team_colour"]
        if col in drivers.columns
    ]
    if driver_cols:
        lap_times = lap_times.merge(
            drivers[driver_cols].drop_duplicates(subset=["driver_number"]),
            on="driver_number",
            how="left",
        )

    if "date_start" in lap_times.columns:
        lap_times["date_start"] = pd.to_datetime(lap_times["date_start"], errors="coerce", utc=True)

    lap_times["team_color"] = lap_times.get("team_colour", pd.Series([None] * len(lap_times))).apply(safe_team_color)
    return lap_times.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)

def enrich_laps_with_stints(laps, stints):
    if laps.empty:
        return pd.DataFrame()

    out = laps.copy()
    if any(col not in out.columns for col in ["driver_number", "lap_number", "lap_duration"]):
        return pd.DataFrame()

    out["driver_number"] = pd.to_numeric(out["driver_number"], errors="coerce")
    out["lap_number"] = pd.to_numeric(out["lap_number"], errors="coerce")
    out["lap_duration"] = pd.to_numeric(out["lap_duration"], errors="coerce")
    out = out.dropna(subset=["driver_number", "lap_number", "lap_duration"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["driver_number"] = out["driver_number"].astype(int)
    out["lap_number"] = out["lap_number"].astype(int)
    out["lap_end"] = pd.to_datetime(out.get("date_start"), errors="coerce", utc=True) + pd.to_timedelta(
        out["lap_duration"], unit="s"
    )

    out["compound"] = "Unknown"
    out["stint_number"] = np.nan
    if stints.empty:
        return out
    needed = ["driver_number", "lap_start", "lap_end"]
    if any(col not in stints.columns for col in needed):
        return out

    st = stints.copy()
    st["driver_number"] = pd.to_numeric(st["driver_number"], errors="coerce")
    st["lap_start"] = pd.to_numeric(st["lap_start"], errors="coerce")
    st["lap_end"] = pd.to_numeric(st["lap_end"], errors="coerce")
    st = st.dropna(subset=["driver_number", "lap_start", "lap_end"]).copy()
    if st.empty:
        return out

    st["driver_number"] = st["driver_number"].astype(int)
    st["lap_start"] = st["lap_start"].astype(int)
    st["lap_end"] = st["lap_end"].astype(int)
    compound_col = "compound" if "compound" in st.columns else ("tyre_compound" if "tyre_compound" in st.columns else None)
    if compound_col is None:
        st["compound"] = "Unknown"
    else:
        st["compound"] = st[compound_col].fillna("Unknown").astype(str)
    if "stint_number" not in st.columns:
        st["stint_number"] = np.nan

    frames = []
    for driver, driver_laps in out.groupby("driver_number"):
        driver_stints = st[st["driver_number"] == driver]
        temp = driver_laps.copy()
        if driver_stints.empty:
            frames.append(temp)
            continue
        for _, sr in driver_stints.iterrows():
            mask = (temp["lap_number"] >= sr["lap_start"]) & (temp["lap_number"] <= sr["lap_end"])
            temp.loc[mask, "compound"] = sr["compound"]
            temp.loc[mask, "stint_number"] = sr["stint_number"]
        frames.append(temp)

    return pd.concat(frames, ignore_index=True)

def derive_stint_windows(stints, laps_enriched):
    if stints.empty:
        return pd.DataFrame()
    if any(col not in stints.columns for col in ["driver_number", "lap_start", "lap_end"]):
        return pd.DataFrame()
    if laps_enriched.empty:
        return pd.DataFrame()

    st = stints.copy()
    st["driver_number"] = pd.to_numeric(st["driver_number"], errors="coerce")
    st["lap_start"] = pd.to_numeric(st["lap_start"], errors="coerce")
    st["lap_end"] = pd.to_numeric(st["lap_end"], errors="coerce")
    st = st.dropna(subset=["driver_number", "lap_start", "lap_end"]).copy()
    if st.empty:
        return pd.DataFrame()
    st["driver_number"] = st["driver_number"].astype(int)
    st["lap_start"] = st["lap_start"].astype(int)
    st["lap_end"] = st["lap_end"].astype(int)

    result = []
    for _, row in st.iterrows():
        mask = (
            (laps_enriched["driver_number"] == row["driver_number"])
            & (laps_enriched["lap_number"] >= row["lap_start"])
            & (laps_enriched["lap_number"] <= row["lap_end"])
            & laps_enriched["lap_end"].notna()
        )
        seg = laps_enriched.loc[mask]
        if seg.empty:
            continue
        item = row.to_dict()
        item["date_start"] = seg["lap_end"].min()
        item["date_end"] = seg["lap_end"].max()
        compound_col = "compound" if "compound" in row else ("tyre_compound" if "tyre_compound" in row else None)
        item["compound_label"] = (
            str(row.get(compound_col, "Unknown")) if compound_col is not None else str(row.get("compound", "Unknown"))
        )
        result.append(item)

    return pd.DataFrame(result)

@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_short)
def fetch_driver_car_data(session_key, driver_number):
    return pd.DataFrame(get_json("car_data", {"session_key": session_key, "driver_number": int(driver_number)}))

