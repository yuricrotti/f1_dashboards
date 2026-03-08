import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import APP_CONFIG, BASE, PLOT_CONFIG, logger


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


def compute_best_laps(laps, drivers):
    if laps.empty:
        raise ValueError("DataFrame laps está vazio.")

    valid = laps.dropna(subset=["lap_duration"]).copy()
    if valid.empty:
        raise ValueError("Nenhuma volta válida com lap_duration encontrada.")

    best_laps = (
        valid.sort_values(["driver_number", "lap_duration"]).groupby("driver_number", as_index=False).first()
    )

    driver_cols = [
        c
        for c in [
            "driver_number",
            "full_name",
            "name_acronym",
            "team_name",
            "team_colour",
        ]
        if c in drivers.columns
    ]

    if driver_cols:
        best_laps = best_laps.merge(
            drivers[driver_cols].drop_duplicates(subset=["driver_number"]),
            on="driver_number",
            how="left",
        )

    desired_cols = [
        "driver_number",
        "full_name",
        "name_acronym",
        "team_name",
        "team_colour",
        "lap_number",
        "lap_duration",
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3",
        "i1_speed",
        "i2_speed",
        "st_speed",
        "date_start",
    ]
    desired_cols = [c for c in desired_cols if c in best_laps.columns]

    best_laps = best_laps[desired_cols].copy()
    best_laps = best_laps.sort_values("lap_duration").reset_index(drop=True)
    best_laps["gap_to_leader"] = best_laps["lap_duration"] - best_laps["lap_duration"].min()
    best_laps["position"] = np.arange(1, len(best_laps) + 1)
    return best_laps


def sector_rankings(best_laps):
    sectors = {
        "S1": "duration_sector_1",
        "S2": "duration_sector_2",
        "S3": "duration_sector_3",
    }

    rankings = {}
    for sector_name, col in sectors.items():
        if col not in best_laps.columns:
            continue

        df = best_laps.dropna(subset=[col]).copy().sort_values(col).reset_index(drop=True)
        if df.empty:
            continue

        best = df.loc[0, col]
        df["gap"] = df[col] - best
        df["rank"] = np.arange(1, len(df) + 1)
        rankings[sector_name] = df

    return rankings


def compute_team_summary(best_laps):
    if best_laps.empty or "team_name" not in best_laps.columns:
        return pd.DataFrame()

    team_summary = (
        best_laps.groupby("team_name", as_index=False)
        .agg(
            best_lap=("lap_duration", "min"),
            avg_lap=("lap_duration", "mean"),
            avg_top_speed=("st_speed", "mean"),
            drivers=("driver_number", "count"),
        )
        .sort_values("best_lap")
        .reset_index(drop=True)
    )
    team_summary["gap_to_best_team"] = team_summary["best_lap"] - team_summary["best_lap"].min()
    return team_summary


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


def create_session_timeline(laps_enriched, stints, race_control, weather, selected_driver_numbers):
    fig = go.Figure()
    has_any = False

    if not laps_enriched.empty:
        lap_df = laps_enriched.copy()
        if selected_driver_numbers:
            lap_df = lap_df[lap_df["driver_number"].isin(selected_driver_numbers)]
        lap_df = lap_df[lap_df["lap_end"].notna()]
        if not lap_df.empty:
            labels = lap_df["name_acronym"] if "name_acronym" in lap_df.columns else lap_df["driver_number"].astype(str)
            fig.add_trace(
                go.Scatter(
                    x=lap_df["lap_end"],
                    y=np.full(len(lap_df), 3),
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=lap_df["lap_duration"],
                        colorscale="Turbo",
                        colorbar=dict(title="Lap (s)"),
                        line=dict(width=0),
                    ),
                    customdata=np.column_stack([labels.astype(str), lap_df["lap_number"].astype(str)]),
                    hovertemplate="<b>%{customdata[0]}</b><br>Volta %{customdata[1]}<br>%{marker.color:.3f}s<extra></extra>",
                    name="Laps",
                )
            )
            has_any = True

    stint_windows = derive_stint_windows(stints, laps_enriched)
    if not stint_windows.empty:
        st_df = stint_windows.copy()
        if selected_driver_numbers:
            st_df = st_df[st_df["driver_number"].isin(selected_driver_numbers)]
        st_df = st_df[st_df["date_start"].notna() & st_df["date_end"].notna()]
        for _, row in st_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["date_start"], row["date_end"]],
                    y=[2, 2],
                    mode="lines",
                    line=dict(width=8),
                    name=f"Stint {int(row['driver_number'])}",
                    legendgroup=f"stint_{int(row['driver_number'])}",
                    showlegend=False,
                    customdata=[[row.get("compound_label", "Unknown")], [row.get("compound_label", "Unknown")]],
                    hovertemplate=(
                        f"Driver {int(row['driver_number'])}<br>"
                        f"Compound: {row.get('compound_label', 'Unknown')}<br>"
                        "Início/Fim: %{x}<extra></extra>"
                    ),
                )
            )
        if not st_df.empty:
            has_any = True

    if not race_control.empty and "date" in race_control.columns:
        rc = race_control.copy()
        rc["date"] = pd.to_datetime(rc["date"], errors="coerce", utc=True)
        rc = rc[rc["date"].notna()]
        if not rc.empty:
            event_col = "message" if "message" in rc.columns else ("category" if "category" in rc.columns else None)
            text_vals = rc[event_col].fillna("Evento") if event_col else pd.Series(["Evento"] * len(rc))
            fig.add_trace(
                go.Scatter(
                    x=rc["date"],
                    y=np.full(len(rc), 1),
                    mode="markers",
                    marker=dict(size=9, color="#EF4444", symbol="diamond"),
                    text=text_vals,
                    hovertemplate="%{text}<br>%{x}<extra></extra>",
                    name="Race control",
                )
            )
            has_any = True

    if not weather.empty and "date" in weather.columns:
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"], errors="coerce", utc=True)
        w = w[w["date"].notna()]
        if not w.empty:
            if "air_temperature" in w.columns:
                fig.add_trace(
                    go.Scatter(
                        x=w["date"],
                        y=w["air_temperature"],
                        mode="lines",
                        line=dict(color="#60A5FA", width=2),
                        name="Ar (°C)",
                        yaxis="y2",
                        hovertemplate="Ar: %{y:.1f}°C<br>%{x}<extra></extra>",
                    )
                )
                has_any = True
            if "track_temperature" in w.columns:
                fig.add_trace(
                    go.Scatter(
                        x=w["date"],
                        y=w["track_temperature"],
                        mode="lines",
                        line=dict(color="#F59E0B", width=2),
                        name="Pista (°C)",
                        yaxis="y2",
                        hovertemplate="Pista: %{y:.1f}°C<br>%{x}<extra></extra>",
                    )
                )
                has_any = True

    if not has_any:
        return None

    fig.update_layout(
        title="Linha do tempo unificada da sessão",
        template="plotly_dark",
        height=520,
        xaxis=dict(title="Tempo"),
        yaxis=dict(
            title="Camadas",
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=["Race control", "Stints", "Laps"],
            range=[0.5, 3.5],
        ),
        yaxis2=dict(
            title="Temperatura (°C)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def create_tyre_analytics_charts(laps_enriched):
    if laps_enriched.empty:
        return None, None, None

    lap_df = laps_enriched[laps_enriched["lap_duration"].notna()].copy()
    if lap_df.empty:
        return None, None, None
    lap_df["compound"] = lap_df.get("compound", pd.Series(["Unknown"] * len(lap_df))).fillna("Unknown").astype(str)

    pace_fig = go.Figure()
    for compound, cdf in lap_df.groupby("compound"):
        pace_fig.add_trace(
            go.Box(
                y=cdf["lap_duration"],
                name=compound,
                boxpoints="outliers",
                hovertemplate=f"{compound}<br>%{{y:.3f}}s<extra></extra>",
            )
        )
    pace_fig.update_layout(
        title="Pace por composto",
        yaxis_title="Lap time (s)",
        yaxis_autorange="reversed",
        template="plotly_dark",
        height=420,
    )

    in_col = "is_pit_out_lap" if "is_pit_out_lap" in lap_df.columns else None
    out_col = "is_pit_in_lap" if "is_pit_in_lap" in lap_df.columns else None
    lap_df["lap_type"] = "push"
    if in_col is not None:
        lap_df.loc[lap_df[in_col] == True, "lap_type"] = "outlap"
    if out_col is not None:
        lap_df.loc[lap_df[out_col] == True, "lap_type"] = "inlap"
        lap_df.loc[(lap_df[out_col] != True) & (lap_df["lap_type"] == "push"), "lap_type"] = "push"

    lap_type_order = ["outlap", "push", "inlap"]
    agg = (
        lap_df.groupby("lap_type", as_index=False)["lap_duration"]
        .mean()
        .set_index("lap_type")
        .reindex(lap_type_order)
        .dropna()
        .reset_index()
    )
    compare_fig = None
    if not agg.empty:
        compare_fig = go.Figure(
            go.Bar(
                x=agg["lap_type"],
                y=agg["lap_duration"],
                text=agg["lap_duration"].map(lambda x: f"{x:.3f}s"),
                textposition="outside",
                marker_color=["#3B82F6", "#22C55E", "#EF4444"][: len(agg)],
                hovertemplate="%{x}: %{y:.3f}s<extra></extra>",
            )
        )
        compare_fig.update_layout(
            title="Comparação inlap/outlap vs push",
            xaxis_title="Tipo de volta",
            yaxis_title="Lap time médio (s)",
            yaxis_autorange="reversed",
            template="plotly_dark",
            height=420,
        )

    degr_fig = None
    if "stint_number" in lap_df.columns:
        degr_records = []
        for (driver, stint), sdf in lap_df.groupby(["driver_number", "stint_number"], dropna=True):
            if len(sdf) < 4:
                continue
            sdf = sdf.sort_values("lap_number")
            x = (sdf["lap_number"] - sdf["lap_number"].min()).to_numpy(dtype=float)
            y = sdf["lap_duration"].to_numpy(dtype=float)
            if np.allclose(x.std(), 0):
                continue
            slope = np.polyfit(x, y, 1)[0]
            degr_records.append(
                {
                    "driver_number": int(driver),
                    "stint_number": int(stint) if pd.notna(stint) else np.nan,
                    "degradation_per_lap": slope,
                    "compound": sdf["compound"].iloc[0] if "compound" in sdf.columns else "Unknown",
                }
            )
        degr_df = pd.DataFrame(degr_records)
        if not degr_df.empty:
            degr_df["label"] = degr_df.apply(
                lambda r: f"D{int(r['driver_number'])}-S{int(r['stint_number'])} ({r['compound']})", axis=1
            )
            degr_fig = go.Figure(
                go.Bar(
                    x=degr_df["label"],
                    y=degr_df["degradation_per_lap"],
                    text=degr_df["degradation_per_lap"].map(lambda x: f"{x:+.3f}s/lap"),
                    textposition="outside",
                    marker_color=np.where(degr_df["degradation_per_lap"] > 0, "#EF4444", "#22C55E"),
                    hovertemplate="%{x}<br>%{y:+.3f}s/lap<extra></extra>",
                )
            )
            degr_fig.update_layout(
                title="Degradação por stint (slope de lap time)",
                xaxis_title="Stints",
                yaxis_title="s/lap",
                template="plotly_dark",
                height=420,
            )
            degr_fig.update_xaxes(tickangle=-35)

    return pace_fig, degr_fig, compare_fig


def _to_bool_flag(series):
    if series is None:
        return pd.Series(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["1", "true", "t", "yes", "y"])


def classify_laps_advanced(laps_enriched):
    if laps_enriched.empty:
        return pd.DataFrame()
    if any(c not in laps_enriched.columns for c in ["driver_number", "lap_number", "lap_duration"]):
        return pd.DataFrame()

    df = laps_enriched.copy()
    df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
    df["lap_number"] = pd.to_numeric(df["lap_number"], errors="coerce")
    df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")
    df = df.dropna(subset=["driver_number", "lap_number", "lap_duration"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["driver_number"] = df["driver_number"].astype(int)
    df["lap_number"] = df["lap_number"].astype(int)

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)

    in_flag = _to_bool_flag(df["is_pit_in_lap"]) if "is_pit_in_lap" in df.columns else pd.Series(False, index=df.index)
    out_flag = _to_bool_flag(df["is_pit_out_lap"]) if "is_pit_out_lap" in df.columns else pd.Series(False, index=df.index)

    stats = df.groupby("driver_number")["lap_duration"].agg(driver_med="median", driver_std="std").reset_index()
    df = df.merge(stats, on="driver_number", how="left")
    df["driver_std"] = df["driver_std"].fillna(0.0)

    traffic_thr = df["driver_med"] + (1.25 * df["driver_std"]).clip(lower=0.7)
    df["lap_phase"] = "push"
    df.loc[out_flag, "lap_phase"] = "outlap"
    df.loc[in_flag, "lap_phase"] = "inlap"
    df.loc[(df["lap_phase"] == "push") & (df["lap_duration"] > traffic_thr), "lap_phase"] = "traffic"

    # smooth push set for strategy analysis
    df["is_clean_push"] = df["lap_phase"] == "push"
    return df


def create_lap_phase_distribution_chart(laps_classified, selected_driver_numbers):
    if laps_classified.empty:
        return None
    df = laps_classified.copy()
    if selected_driver_numbers:
        df = df[df["driver_number"].isin(selected_driver_numbers)]
    if df.empty:
        return None

    phase_order = ["push", "traffic", "outlap", "inlap"]
    agg = (
        df.groupby("lap_phase", as_index=False)["lap_number"]
        .count()
        .rename(columns={"lap_number": "count"})
        .set_index("lap_phase")
        .reindex(phase_order)
        .dropna()
        .reset_index()
    )
    if agg.empty:
        return None

    colors = {"push": "#22C55E", "traffic": "#F97316", "outlap": "#3B82F6", "inlap": "#EF4444"}
    fig = go.Figure(
        go.Bar(
            x=agg["lap_phase"],
            y=agg["count"],
            marker_color=[colors.get(x, "#94A3B8") for x in agg["lap_phase"]],
            text=agg["count"],
            textposition="outside",
            hovertemplate="%{x}: %{y} voltas<extra></extra>",
        )
    )
    fig.update_layout(
        title="Classificação automática de voltas",
        xaxis_title="Classe",
        yaxis_title="Quantidade de voltas",
        template="plotly_dark",
        height=380,
    )
    return fig


def compute_long_run_summary(laps_classified):
    if laps_classified.empty:
        return pd.DataFrame()
    if "stint_number" not in laps_classified.columns:
        return pd.DataFrame()

    df = laps_classified.copy()
    df = df[df["is_clean_push"] == True]
    df = df[df["stint_number"].notna()]
    if df.empty:
        return pd.DataFrame()

    records = []
    for (driver, stint), g in df.groupby(["driver_number", "stint_number"], dropna=True):
        g = g.sort_values("lap_number")
        if len(g) < 5:
            continue
        x = (g["lap_number"] - g["lap_number"].min()).to_numpy(dtype=float)
        y = g["lap_duration"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0] if np.std(x) > 0 else 0.0
        records.append(
            {
                "driver_number": int(driver),
                "stint_number": int(stint),
                "laps": int(len(g)),
                "avg_pace": float(np.mean(y)),
                "consistency_std": float(np.std(y)),
                "degradation_s_per_lap": float(slope),
                "compound": str(g["compound"].iloc[0]) if "compound" in g.columns else "Unknown",
                "driver_label": (
                    str(g["name_acronym"].iloc[0])
                    if "name_acronym" in g.columns and pd.notna(g["name_acronym"].iloc[0])
                    else str(g.get("full_name", pd.Series([driver])).iloc[0])
                ),
                "team_name": str(g["team_name"].iloc[0]) if "team_name" in g.columns else "N/A",
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.sort_values(["avg_pace", "consistency_std"]).reset_index(drop=True)
    return out


def create_long_run_chart(long_run_df):
    if long_run_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=long_run_df["degradation_s_per_lap"],
            y=long_run_df["avg_pace"],
            mode="markers+text",
            text=long_run_df["driver_label"],
            textposition="top center",
            marker=dict(
                size=(long_run_df["laps"].clip(lower=5) * 1.2),
                color=long_run_df["consistency_std"],
                colorscale="Turbo",
                colorbar=dict(title="Std (s)"),
                line=dict(color="white", width=1),
            ),
            customdata=np.column_stack([long_run_df["compound"], long_run_df["team_name"], long_run_df["laps"]]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Equipe: %{customdata[1]}<br>"
                "Composto: %{customdata[0]}<br>"
                "Laps: %{customdata[2]}<br>"
                "Degradação: %{x:+.3f}s/lap<br>"
                "Pace médio: %{y:.3f}s<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Long-run analytics por stint (apenas push laps)",
        xaxis_title="Degradação (s/lap)",
        yaxis_title="Pace médio (s)",
        yaxis_autorange="reversed",
        template="plotly_dark",
        height=430,
    )
    return fig


def build_scorecards(best_laps, laps_classified, long_run_df):
    if best_laps.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = best_laps.copy()
    cols = [c for c in ["driver_number", "full_name", "name_acronym", "team_name", "lap_duration", "gap_to_leader"] if c in base.columns]
    base = base[cols].copy()

    pace_min, pace_max = base["gap_to_leader"].min(), base["gap_to_leader"].max()
    denom_pace = (pace_max - pace_min) if (pace_max - pace_min) > 0 else 1.0
    base["pace_score"] = 100 * (1 - (base["gap_to_leader"] - pace_min) / denom_pace)

    consistency_df = pd.DataFrame(columns=["driver_number", "consistency_std"])
    execution_df = pd.DataFrame(columns=["driver_number", "push_ratio"])
    if not laps_classified.empty:
        tmp = laps_classified.copy()
        consistency_df = (
            tmp[tmp["lap_phase"] == "push"]
            .groupby("driver_number", as_index=False)["lap_duration"]
            .std()
            .rename(columns={"lap_duration": "consistency_std"})
        )
        execution_df = (
            tmp.groupby("driver_number", as_index=False)
            .agg(total_laps=("lap_number", "count"), push_laps=("is_clean_push", "sum"))
        )
        execution_df["push_ratio"] = execution_df["push_laps"] / execution_df["total_laps"].replace(0, np.nan)
        execution_df = execution_df[["driver_number", "push_ratio"]]

    base = base.merge(consistency_df, on="driver_number", how="left")
    base = base.merge(execution_df, on="driver_number", how="left")

    if not long_run_df.empty:
        degr = long_run_df.groupby("driver_number", as_index=False)["degradation_s_per_lap"].mean()
        base = base.merge(degr, on="driver_number", how="left")
    else:
        base["degradation_s_per_lap"] = np.nan

    # Normalize consistency: lower std is better
    c = base["consistency_std"].fillna(base["consistency_std"].median() if base["consistency_std"].notna().any() else 0.2)
    cmin, cmax = c.min(), c.max()
    cden = (cmax - cmin) if (cmax - cmin) > 0 else 1.0
    base["consistency_score"] = 100 * (1 - (c - cmin) / cden)

    # Normalize degradation: lower slope is better
    d = base["degradation_s_per_lap"].fillna(base["degradation_s_per_lap"].median() if base["degradation_s_per_lap"].notna().any() else 0.05)
    dmin, dmax = d.min(), d.max()
    dden = (dmax - dmin) if (dmax - dmin) > 0 else 1.0
    base["degradation_score"] = 100 * (1 - (d - dmin) / dden)

    r = base["push_ratio"].fillna(base["push_ratio"].median() if base["push_ratio"].notna().any() else 0.5)
    base["execution_score"] = (r.clip(0, 1) * 100)

    base["overall_score"] = (
        0.40 * base["pace_score"]
        + 0.25 * base["consistency_score"]
        + 0.20 * base["degradation_score"]
        + 0.15 * base["execution_score"]
    )
    base["overall_score"] = base["overall_score"].round(1)

    driver_score = base.sort_values("overall_score", ascending=False).reset_index(drop=True)

    team_score = pd.DataFrame()
    if "team_name" in driver_score.columns:
        team_score = (
            driver_score.groupby("team_name", as_index=False)
            .agg(
                team_score=("overall_score", "mean"),
                avg_pace_score=("pace_score", "mean"),
                avg_consistency=("consistency_score", "mean"),
                drivers=("driver_number", "count"),
            )
            .sort_values("team_score", ascending=False)
            .reset_index(drop=True)
        )
        team_score["team_score"] = team_score["team_score"].round(1)

    return driver_score, team_score


def create_scorecard_charts(driver_score, team_score):
    driver_fig = None
    team_fig = None

    if not driver_score.empty:
        labels = (
            driver_score["name_acronym"].fillna(driver_score["full_name"]).astype(str)
            if "name_acronym" in driver_score.columns
            else driver_score["full_name"].astype(str)
        )
        driver_fig = go.Figure(
            go.Bar(
                x=labels,
                y=driver_score["overall_score"],
                marker_color="#22D3EE",
                text=driver_score["overall_score"].map(lambda x: f"{x:.1f}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            )
        )
        driver_fig.update_layout(
            title="Scorecard executivo por piloto",
            xaxis_title="Piloto",
            yaxis_title="Score (0-100)",
            template="plotly_dark",
            height=420,
        )
        driver_fig.update_xaxes(tickangle=-35)

    if not team_score.empty:
        team_fig = go.Figure(
            go.Bar(
                x=team_score["team_name"],
                y=team_score["team_score"],
                marker_color="#34D399",
                text=team_score["team_score"].map(lambda x: f"{x:.1f}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            )
        )
        team_fig.update_layout(
            title="Scorecard executivo por equipe",
            xaxis_title="Equipe",
            yaxis_title="Score (0-100)",
            template="plotly_dark",
            height=420,
        )
        team_fig.update_xaxes(tickangle=-25)

    return driver_fig, team_fig


def prepare_teammate_metrics(best_laps, laps_classified):
    if best_laps.empty or "team_name" not in best_laps.columns:
        return pd.DataFrame()

    base_cols = [
        c
        for c in [
            "driver_number",
            "full_name",
            "name_acronym",
            "team_name",
            "lap_duration",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
        ]
        if c in best_laps.columns
    ]
    base = best_laps[base_cols].copy()

    if laps_classified.empty:
        base["consistency_std"] = np.nan
        base["push_ratio"] = np.nan
        return base

    temp = laps_classified.copy()
    cons = (
        temp[temp.get("lap_phase", "push") == "push"]
        .groupby("driver_number", as_index=False)["lap_duration"]
        .std()
        .rename(columns={"lap_duration": "consistency_std"})
    )
    execu = (
        temp.groupby("driver_number", as_index=False)
        .agg(total_laps=("lap_number", "count"), push_laps=("is_clean_push", "sum"))
    )
    execu["push_ratio"] = execu["push_laps"] / execu["total_laps"].replace(0, np.nan)
    execu = execu[["driver_number", "push_ratio"]]
    return base.merge(cons, on="driver_number", how="left").merge(execu, on="driver_number", how="left")


def create_teammate_timing_chart(team_df):
    if team_df.empty or len(team_df) < 2:
        return None
    metrics = [m for m in ["lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3"] if m in team_df.columns]
    if not metrics:
        return None

    labels = (
        team_df["name_acronym"].fillna(team_df["full_name"]).astype(str)
        if "name_acronym" in team_df.columns
        else team_df["full_name"].astype(str)
    )
    fig = go.Figure()
    for idx, row in team_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[m.replace("duration_", "").replace("_", " ").upper() for m in metrics],
                y=[row[m] for m in metrics],
                name=labels.iloc[idx],
                text=[f"{row[m]:.3f}" if pd.notna(row[m]) else "-" for m in metrics],
                textposition="outside",
                hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f}s<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        title="Teammate: tempos por volta e setores",
        yaxis_title="Tempo (s)",
        template="plotly_dark",
        height=360,
    )
    return fig


def create_teammate_ops_chart(team_df):
    if team_df.empty or len(team_df) < 2:
        return None
    if "consistency_std" not in team_df.columns and "push_ratio" not in team_df.columns:
        return None

    labels = (
        team_df["name_acronym"].fillna(team_df["full_name"]).astype(str)
        if "name_acronym" in team_df.columns
        else team_df["full_name"].astype(str)
    )

    fig = go.Figure()
    if "consistency_std" in team_df.columns:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=team_df["consistency_std"],
                name="Consistência (std, menor melhor)",
                marker_color="#38BDF8",
                hovertemplate="%{x}<br>Std: %{y:.3f}s<extra></extra>",
            )
        )
    if "push_ratio" in team_df.columns:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=(team_df["push_ratio"] * 100),
                mode="markers+lines",
                name="Execução (push ratio, maior melhor)",
                marker=dict(size=10, color="#F59E0B"),
                yaxis="y2",
                hovertemplate="%{x}<br>Push ratio: %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="Teammate: consistência e execução",
        template="plotly_dark",
        height=360,
        yaxis=dict(title="Std (s)"),
        yaxis2=dict(title="Push ratio (%)", overlaying="y", side="right", showgrid=False),
    )
    return fig


def build_teammate_summary(team_df):
    if team_df.empty or len(team_df) < 2:
        return []
    d1, d2 = team_df.iloc[0], team_df.iloc[1]
    n1 = d1.get("name_acronym") or d1.get("full_name", "D1")
    n2 = d2.get("name_acronym") or d2.get("full_name", "D2")
    out = []

    if pd.notna(d1.get("lap_duration")) and pd.notna(d2.get("lap_duration")):
        winner = n1 if d1["lap_duration"] < d2["lap_duration"] else n2
        gap = abs(float(d1["lap_duration"] - d2["lap_duration"]))
        out.append(f"Ritmo de volta: {winner} à frente por {gap:.3f}s.")

    for sec, label in [("duration_sector_1", "S1"), ("duration_sector_2", "S2"), ("duration_sector_3", "S3")]:
        if sec in team_df.columns and pd.notna(d1.get(sec)) and pd.notna(d2.get(sec)):
            winner = n1 if d1[sec] < d2[sec] else n2
            gap = abs(float(d1[sec] - d2[sec]))
            out.append(f"{label}: vantagem de {winner} por {gap:.3f}s.")

    if "consistency_std" in team_df.columns and pd.notna(d1.get("consistency_std")) and pd.notna(d2.get("consistency_std")):
        winner = n1 if d1["consistency_std"] < d2["consistency_std"] else n2
        out.append(f"Consistência: {winner} apresenta menor variância de ritmo.")

    if "push_ratio" in team_df.columns and pd.notna(d1.get("push_ratio")) and pd.notna(d2.get("push_ratio")):
        winner = n1 if d1["push_ratio"] > d2["push_ratio"] else n2
        out.append(f"Execução: {winner} tem maior proporção de voltas limpas push.")

    return out[:6]


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_short)
def fetch_driver_car_data(session_key, driver_number):
    return pd.DataFrame(get_json("car_data", {"session_key": session_key, "driver_number": int(driver_number)}))


def build_driver_delta_trace(laps, location, position, session_key, driver_number):
    if laps.empty:
        return None

    driver_laps = laps.copy()
    driver_laps = driver_laps[pd.to_numeric(driver_laps.get("driver_number"), errors="coerce") == int(driver_number)]
    if driver_laps.empty:
        return None
    if "lap_duration" not in driver_laps.columns or "date_start" not in driver_laps.columns:
        return None

    driver_laps["lap_duration"] = pd.to_numeric(driver_laps["lap_duration"], errors="coerce")
    driver_laps["date_start"] = pd.to_datetime(driver_laps["date_start"], errors="coerce", utc=True)
    driver_laps = driver_laps.dropna(subset=["lap_duration", "date_start"]).sort_values("lap_duration")
    if driver_laps.empty:
        return None

    best = driver_laps.iloc[0]
    start = best["date_start"]
    end = start + pd.to_timedelta(float(best["lap_duration"]), unit="s")

    source_df = location if not location.empty else position
    if source_df.empty:
        return None
    norm, _ = _normalize_track_xy(source_df)
    if norm.empty or "date" not in norm.columns:
        return None

    norm["driver_number"] = pd.to_numeric(norm["driver_number"], errors="coerce")
    norm["date"] = pd.to_datetime(norm["date"], errors="coerce", utc=True)
    track = norm[
        (norm["driver_number"] == int(driver_number))
        & norm["date"].notna()
        & (norm["date"] >= start)
        & (norm["date"] <= end)
    ][["date", "x", "y"]].copy()
    if len(track) < 15:
        return None
    track = track.sort_values("date")

    car = fetch_driver_car_data(session_key, int(driver_number))
    speed_col = "speed" if "speed" in car.columns else ("speed_kmh" if "speed_kmh" in car.columns else None)
    if speed_col and "date" in car.columns:
        car["date"] = pd.to_datetime(car["date"], errors="coerce", utc=True)
        car[speed_col] = pd.to_numeric(car[speed_col], errors="coerce")
        car = car[car["date"].notna() & car[speed_col].notna()].sort_values("date")
        if not car.empty:
            track = pd.merge_asof(
                track.sort_values("date"),
                car[["date", speed_col]].sort_values("date"),
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta("500ms"),
            )
            track = track.rename(columns={speed_col: "speed"})

    dx = track["x"].diff().fillna(0.0)
    dy = track["y"].diff().fillna(0.0)
    dist = np.sqrt(dx * dx + dy * dy).cumsum()
    total = float(dist.iloc[-1]) if len(dist) else 0.0
    if total <= 0:
        return None

    track["progress"] = dist / total
    track["elapsed"] = (track["date"] - start).dt.total_seconds()
    trace = track[["progress", "elapsed"]].dropna().drop_duplicates(subset=["progress"]).sort_values("progress")
    if len(trace) < 10:
        return None
    return trace


def create_telemetry_delta_chart(trace_a, trace_b, label_a, label_b):
    if trace_a is None or trace_b is None:
        return None

    grid = np.linspace(0, 1, 250)
    a = np.interp(grid, trace_a["progress"].to_numpy(), trace_a["elapsed"].to_numpy())
    b = np.interp(grid, trace_b["progress"].to_numpy(), trace_b["elapsed"].to_numpy())
    delta = b - a

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid * 100,
            y=delta,
            mode="lines",
            line=dict(color="#38BDF8", width=3),
            name="Delta acumulado",
            hovertemplate="Trecho: %{x:.1f}%<br>Delta: %{y:+.3f}s<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    fig.update_layout(
        title=f"Delta acumulado por trecho da pista ({label_b} vs {label_a})",
        xaxis_title="Progresso no traçado (%)",
        yaxis_title="Delta acumulado (s)",
        template="plotly_dark",
        height=420,
    )
    return fig


def _normalize_track_xy(df):
    if df.empty:
        return pd.DataFrame(), None

    candidates = [("x", "y"), ("position_x", "position_y")]
    chosen = None
    for x_col, y_col in candidates:
        if x_col in df.columns and y_col in df.columns:
            chosen = (x_col, y_col)
            break

    if chosen is None or "driver_number" not in df.columns:
        return pd.DataFrame(), None

    x_col, y_col = chosen
    out = df.copy()
    out = out.rename(columns={x_col: "x", y_col: "y"})
    return out, f"{x_col}/{y_col}"


def create_driver_comparison_chart(best_laps, selected_driver_numbers):
    if not selected_driver_numbers:
        return None

    comp = best_laps[best_laps["driver_number"].isin(selected_driver_numbers)].copy()
    if comp.empty:
        return None

    metric_cols = ["lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3"]
    metric_cols = [col for col in metric_cols if col in comp.columns]
    if not metric_cols:
        return None

    labels = comp["name_acronym"] if "name_acronym" in comp.columns else comp["full_name"]
    comp["driver_label"] = labels.fillna(comp["driver_number"].astype(str))

    metric_names = {
        "lap_duration": "Lap",
        "duration_sector_1": "S1",
        "duration_sector_2": "S2",
        "duration_sector_3": "S3",
    }
    x_axis = [metric_names[m] for m in metric_cols]

    fig = go.Figure()
    for _, row in comp.iterrows():
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=[row[m] for m in metric_cols],
                name=row["driver_label"],
                marker_color=safe_team_color(row.get("team_colour")),
                text=[f"{row[m]:.3f}" if pd.notna(row[m]) else "-" for m in metric_cols],
                textposition="outside",
                hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f}s<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        height=450,
        title="Comparação direta de pilotos (melhor volta e setores)",
        yaxis_title="Tempo (s)",
        legend_title="Pilotos",
        template="plotly_dark",
    )
    return fig


def create_lap_evolution_chart(lap_times, selected_driver_numbers):
    if lap_times.empty or not selected_driver_numbers:
        return None

    df = lap_times[
        (lap_times["driver_number"].isin(selected_driver_numbers)) & lap_times["lap_duration"].notna()
    ].copy()
    if df.empty:
        return None

    labels = df["name_acronym"] if "name_acronym" in df.columns else df["full_name"]
    df["driver_label"] = labels.fillna(df["driver_number"].astype(str))

    fig = go.Figure()
    for driver_number in selected_driver_numbers:
        driver_df = df[df["driver_number"] == driver_number].sort_values("lap_number")
        if driver_df.empty:
            continue
        label = driver_df["driver_label"].iloc[0]
        color = safe_team_color(driver_df["team_colour"].iloc[0] if "team_colour" in driver_df.columns else None)
        fig.add_trace(
            go.Scatter(
                x=driver_df["lap_number"],
                y=driver_df["lap_duration"],
                mode="lines+markers",
                name=label,
                marker=dict(size=6, color=color),
                line=dict(width=2, color=color),
                hovertemplate="<b>%{fullData.name}</b><br>Volta %{x}<br>%{y:.3f}s<extra></extra>",
            )
        )

    fig.update_layout(
        title="Evolução de ritmo por volta",
        xaxis_title="Número da volta",
        yaxis_title="Lap time (s)",
        yaxis_autorange="reversed",
        height=460,
        template="plotly_dark",
    )
    return fig


def create_consistency_boxplot(lap_times, selected_driver_numbers):
    if lap_times.empty or not selected_driver_numbers:
        return None

    df = lap_times[
        (lap_times["driver_number"].isin(selected_driver_numbers)) & lap_times["lap_duration"].notna()
    ].copy()
    if df.empty:
        return None

    labels = df["name_acronym"] if "name_acronym" in df.columns else df["full_name"]
    df["driver_label"] = labels.fillna(df["driver_number"].astype(str))

    fig = go.Figure()
    for driver_number in selected_driver_numbers:
        driver_df = df[df["driver_number"] == driver_number]
        if driver_df.empty:
            continue
        label = driver_df["driver_label"].iloc[0]
        color = safe_team_color(driver_df["team_colour"].iloc[0] if "team_colour" in driver_df.columns else None)
        fig.add_trace(
            go.Box(
                y=driver_df["lap_duration"],
                name=label,
                boxpoints="outliers",
                marker_color=color,
                line=dict(color=color),
                hovertemplate="<b>%{fullData.name}</b><br>%{y:.3f}s<extra></extra>",
            )
        )

    fig.update_layout(
        title="Consistência de ritmo (distribuição das voltas)",
        yaxis_title="Lap time (s)",
        yaxis_autorange="reversed",
        height=460,
        template="plotly_dark",
    )
    return fig


def create_race_position_evolution_chart(
    position_df,
    laps_df,
    drivers,
    selected_driver_numbers=None,
    only_finishers=False,
    completion_ratio=0.9,
):
    if position_df.empty or laps_df.empty:
        return None
    if any(c not in position_df.columns for c in ["date", "driver_number", "position"]):
        return None
    if any(c not in laps_df.columns for c in ["driver_number", "lap_number", "date_start", "lap_duration"]):
        return None

    pos = position_df.copy()
    pos["date"] = pd.to_datetime(pos["date"], errors="coerce", utc=True)
    pos["driver_number"] = pd.to_numeric(pos["driver_number"], errors="coerce")
    pos["position"] = pd.to_numeric(pos["position"], errors="coerce")
    pos = pos.dropna(subset=["date", "driver_number", "position"]).copy()
    if pos.empty:
        return None
    pos["driver_number"] = pos["driver_number"].astype(int)

    laps = laps_df.copy()
    laps["driver_number"] = pd.to_numeric(laps["driver_number"], errors="coerce")
    laps["lap_number"] = pd.to_numeric(laps["lap_number"], errors="coerce")
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps["date_start"] = pd.to_datetime(laps["date_start"], errors="coerce", utc=True)
    laps = laps.dropna(subset=["driver_number", "lap_number", "lap_duration", "date_start"]).copy()
    if laps.empty:
        return None
    laps["driver_number"] = laps["driver_number"].astype(int)
    laps["lap_number"] = laps["lap_number"].astype(int)
    laps["timestamp"] = laps["date_start"] + pd.to_timedelta(laps["lap_duration"], unit="s")

    if selected_driver_numbers:
        selected = set(int(x) for x in selected_driver_numbers)
        pos = pos[pos["driver_number"].isin(selected)]
        laps = laps[laps["driver_number"].isin(selected)]
    if pos.empty or laps.empty:
        return None

    pos_key = pos.rename(columns={"date": "timestamp"})[["driver_number", "timestamp", "position"]].copy()
    laps_key = laps[["driver_number", "lap_number", "timestamp"]].copy()

    merged_parts = []
    common_drivers = sorted(set(laps_key["driver_number"].unique()).intersection(set(pos_key["driver_number"].unique())))
    for drv in common_drivers:
        ldf = laps_key[laps_key["driver_number"] == drv].sort_values("timestamp")
        pdf = pos_key[pos_key["driver_number"] == drv].sort_values("timestamp")
        if ldf.empty or pdf.empty:
            continue
        part = pd.merge_asof(
            ldf,
            pdf[["timestamp", "position"]],
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("30s"),
        )
        part["driver_number"] = int(drv)
        merged_parts.append(part)

    if not merged_parts:
        return None

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.dropna(subset=["position"]).copy()
    if merged.empty:
        return None

    if only_finishers:
        global_max_lap = int(merged["lap_number"].max())
        min_lap_threshold = max(1, int(global_max_lap * float(completion_ratio)))
        lap_count = merged.groupby("driver_number", as_index=False)["lap_number"].max()
        finishers = lap_count[lap_count["lap_number"] >= min_lap_threshold]["driver_number"].tolist()
        merged = merged[merged["driver_number"].isin(finishers)].copy()
        if merged.empty:
            return None

    if not drivers.empty and "driver_number" in drivers.columns:
        dcols = [c for c in ["driver_number", "name_acronym", "full_name", "team_colour"] if c in drivers.columns]
        if dcols:
            dmeta = drivers[dcols].drop_duplicates(subset=["driver_number"]).copy()
            dmeta["driver_number"] = pd.to_numeric(dmeta["driver_number"], errors="coerce")
            merged = merged.merge(dmeta, on="driver_number", how="left")

    merged["driver_label"] = (
        merged["name_acronym"].fillna(merged.get("full_name", pd.Series(["N/A"] * len(merged)))).astype(str)
        if "name_acronym" in merged.columns
        else merged["driver_number"].astype(str)
    )
    merged["team_color"] = merged.get("team_colour", pd.Series([None] * len(merged))).apply(safe_team_color)

    # Expand each driver to every lap and forward-fill position so lines reach the final lap.
    driver_last_lap = laps_key.groupby("driver_number", as_index=False)["lap_number"].max()
    expanded_parts = []
    for driver in sorted(merged["driver_number"].dropna().unique()):
        ddf = merged[merged["driver_number"] == driver].sort_values("lap_number").copy()
        if ddf.empty:
            continue
        last_lap_row = driver_last_lap[driver_last_lap["driver_number"] == driver]
        if last_lap_row.empty:
            max_lap = int(ddf["lap_number"].max())
        else:
            max_lap = int(last_lap_row["lap_number"].iloc[0])
        max_lap = max(max_lap, int(ddf["lap_number"].max()))

        full_laps = pd.DataFrame({"lap_number": np.arange(1, max_lap + 1, dtype=int)})
        base_cols = ["lap_number", "position", "driver_number", "driver_label", "team_color"]
        ddf_base = ddf[[c for c in base_cols if c in ddf.columns]].drop_duplicates(subset=["lap_number"])
        ddf_full = full_laps.merge(ddf_base, on="lap_number", how="left")
        ddf_full["driver_number"] = int(driver)
        ddf_full["driver_label"] = ddf["driver_label"].iloc[0]
        ddf_full["team_color"] = ddf["team_color"].iloc[0]
        ddf_full["position"] = ddf_full["position"].ffill().bfill()
        ddf_full = ddf_full.dropna(subset=["position"])
        if not ddf_full.empty:
            expanded_parts.append(ddf_full)

    if expanded_parts:
        merged = pd.concat(expanded_parts, ignore_index=True)
    if merged.empty:
        return None

    fig = go.Figure()
    for driver in sorted(merged["driver_number"].dropna().unique()):
        ddf = merged[merged["driver_number"] == driver].sort_values("lap_number")
        if ddf.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=ddf["lap_number"],
                y=ddf["position"],
                mode="lines+markers",
                name=ddf["driver_label"].iloc[0],
                line=dict(width=2.2, color=ddf["team_color"].iloc[0]),
                marker=dict(size=5, color=ddf["team_color"].iloc[0]),
                hovertemplate="<b>%{fullData.name}</b><br>Volta %{x}<br>Posição P%{y:.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Evolução de posição por volta (corrida)",
        template="plotly_dark",
        height=520,
        xaxis_title="Volta",
        yaxis_title="Posição",
        legend_title="Pilotos",
    )
    fig.update_yaxes(autorange="reversed", dtick=1)
    return fig


def create_sector_delta_heatmap(best_laps):
    sector_cols = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
    if best_laps.empty or any(col not in best_laps.columns for col in sector_cols):
        return None

    df = best_laps.dropna(subset=sector_cols).copy()
    if df.empty:
        return None

    leader = df.iloc[0]
    labels = df["name_acronym"] if "name_acronym" in df.columns else df["full_name"]
    y_labels = labels.fillna(df["driver_number"].astype(str)).tolist()

    z = np.column_stack(
        [
            (df["duration_sector_1"] - leader["duration_sector_1"]).to_numpy(),
            (df["duration_sector_2"] - leader["duration_sector_2"]).to_numpy(),
            (df["duration_sector_3"] - leader["duration_sector_3"]).to_numpy(),
        ]
    )
    zmax = float(np.nanmax(z)) if np.isfinite(np.nanmax(z)) else 0.1
    if zmax <= 0:
        zmax = 0.1

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["S1", "S2", "S3"],
            y=y_labels,
            zmin=0,
            zmax=zmax,
            colorscale=[[0, "#22C55E"], [0.5, "#F59E0B"], [1, "#EF4444"]],
            colorbar=dict(title="Delta (s)"),
            hovertemplate="<b>%{y}</b><br>%{x}: +%{z:.3f}s<extra></extra>",
        )
    )

    fig.update_layout(
        title="Heatmap de delta setorial vs líder da sessão",
        xaxis_title="Setor",
        yaxis_title="Piloto",
        height=560,
        template="plotly_dark",
    )
    return fig


def create_sector_3d_chart(best_laps):
    sector_cols = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
    if best_laps.empty or any(col not in best_laps.columns for col in sector_cols):
        return None

    df = best_laps.dropna(subset=sector_cols).copy()
    if df.empty:
        return None

    df["driver_label"] = (
        df["name_acronym"].fillna(df.get("full_name", pd.Series(["N/A"] * len(df)))).astype(str)
        if "name_acronym" in df.columns
        else df.get("full_name", pd.Series(["N/A"] * len(df))).astype(str)
    )
    df["team_color"] = df.get("team_colour", pd.Series([None] * len(df))).apply(safe_team_color)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["duration_sector_1"],
                y=df["duration_sector_2"],
                z=df["duration_sector_3"],
                mode="markers+text",
                text=df["driver_label"],
                textposition="top center",
                marker=dict(
                    size=7,
                    color=df["team_color"],
                    line=dict(color="white", width=1),
                    opacity=0.92,
                ),
                customdata=np.column_stack(
                    [
                        df.get("full_name", df["driver_label"]).astype(str),
                        df.get("team_name", pd.Series(["N/A"] * len(df))).astype(str),
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Equipe: %{customdata[1]}<br>"
                    "S1: %{x:.3f}s<br>"
                    "S2: %{y:.3f}s<br>"
                    "S3: %{z:.3f}s<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="Mapa setorial 3D (S1 x S2 x S3)",
        template="plotly_dark",
        height=640,
        scene=dict(
            xaxis_title="Sector 1 (s)",
            yaxis_title="Sector 2 (s)",
            zaxis_title="Sector 3 (s)",
            camera=dict(eye=dict(x=1.45, y=1.3, z=1.0)),
        ),
    )
    return fig


def build_insights(best_laps, rankings):
    insights = []

    leader = best_laps.iloc[0]
    insights.append(f"{leader['full_name']} foi o mais rápido da sessão com {leader['lap_duration']:.3f}s.")

    if "st_speed" in best_laps.columns and best_laps["st_speed"].notna().any():
        fastest_speed_row = best_laps.loc[best_laps["st_speed"].idxmax()]
        insights.append(
            f"{fastest_speed_row['full_name']} registrou a maior speed trap: {fastest_speed_row['st_speed']:.0f} km/h."
        )

        if fastest_speed_row["full_name"] != leader["full_name"]:
            insights.append("Maior velocidade final não significou necessariamente melhor volta.")

    if "S1" in rankings and not rankings["S1"].empty:
        s1_leader = rankings["S1"].iloc[0]
        insights.append(f"S1: {s1_leader['full_name']} foi a referência do setor.")

    if "S2" in rankings and not rankings["S2"].empty:
        s2_leader = rankings["S2"].iloc[0]
        insights.append(f"S2: {s2_leader['full_name']} liderou o setor.")

    if "S3" in rankings and not rankings["S3"].empty:
        s3_leader = rankings["S3"].iloc[0]
        insights.append(f"S3: {s3_leader['full_name']} foi a melhor referência.")

    return insights[:5]


def build_marketing_text(best_laps, rankings, country, year, session_name):
    leader = best_laps.iloc[0]

    s1 = rankings["S1"].iloc[0]["team_name"] if "S1" in rankings and not rankings["S1"].empty else "N/A"
    s2 = rankings["S2"].iloc[0]["team_name"] if "S2" in rankings and not rankings["S2"].empty else "N/A"
    s3 = rankings["S3"].iloc[0]["team_name"] if "S3" in rankings and not rankings["S3"].empty else "N/A"

    lines = [
        "===== TEXTO PRONTO PARA POST =====",
        f"{country} GP {year} • {session_name}",
        f"{leader['full_name']} liderou a sessão com {leader['lap_duration']:.3f}s.",
        f"Setores: S1={s1} | S2={s2} | S3={s3}.",
        "Leitura principal: velocidade final isolada não explica o ritmo total de volta.",
        "O desempenho aponta para equilíbrio entre eficiência aerodinâmica, tração e gestão do setup.",
    ]
    return "\n".join(lines)


def create_professional_infographic(best_laps, rankings, team_summary, session, session_type=None, race_winner=None):
    best_laps = best_laps.copy()
    is_race = session_type == "race"

    bg = "#0B1220"
    panel = "#111827"
    grid = "rgba(255,255,255,0.08)"
    font_color = "#E5E7EB"
    sub_font = "#AAB6C3"

    def team_color(row):
        return safe_team_color(row.get("team_colour", None))

    best_laps["team_color"] = best_laps.apply(team_color, axis=1)

    rng = np.random.default_rng(42)
    if "st_speed" in best_laps.columns:
        st_speed_series = best_laps["st_speed"].fillna(best_laps["st_speed"].median())
    else:
        st_speed_series = pd.Series(np.zeros(len(best_laps)))
    best_laps["st_speed_plot"] = st_speed_series + rng.uniform(-0.8, 0.8, len(best_laps))
    best_laps["lap_duration_plot"] = best_laps["lap_duration"] + rng.uniform(-0.015, 0.015, len(best_laps))

    fig = make_subplots(
        rows=3,
        cols=2,
        row_heights=[0.38, 0.31, 0.31],
        column_widths=[0.62, 0.38],
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        subplot_titles=(
            "Top Speed vs Lap Time",
            "Top 10 da sessão" if not is_race else "Top 10 Fastest Laps",
            "Gap para o líder",
            "Ranking por setor (Top 5)",
            "Top Teams" if not is_race else "Top Teams (Fastest Lap)",
            "Mapa de desempenho",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    customdata_scatter = np.column_stack([
        best_laps.get("full_name", pd.Series(["N/A"] * len(best_laps))).astype(str),
        best_laps.get("team_name", pd.Series(["N/A"] * len(best_laps))).astype(str),
        best_laps["lap_duration"].round(3).astype(str),
        best_laps.get("st_speed", pd.Series([0] * len(best_laps))).fillna(0).round(0).astype(int).astype(str),
        best_laps["gap_to_leader"].round(3).astype(str),
        (
            best_laps["name_acronym"].astype(str)
            if "name_acronym" in best_laps.columns
            else best_laps.get("full_name", pd.Series(["N/A"] * len(best_laps))).astype(str)
        ),
    ])

    fig.add_trace(
        go.Scatter(
            x=best_laps["st_speed_plot"],
            y=best_laps["lap_duration_plot"],
            mode="markers",
            marker=dict(
                size=16,
                color=best_laps["team_color"],
                line=dict(color="white", width=1.2),
            ),
            customdata=customdata_scatter,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Sigla: %{customdata[5]}<br>"
                "Equipe: %{customdata[1]}<br>"
                "Lap Time: %{customdata[2]}s<br>"
                "Top Speed: %{customdata[3]} km/h<br>"
                "Gap: %{customdata[4]}s<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    top5 = best_laps.head(5)
    fig.add_trace(
        go.Scatter(
            x=top5["st_speed_plot"],
            y=top5["lap_duration_plot"],
            mode="text",
            text=top5["name_acronym"] if "name_acronym" in top5.columns else top5["full_name"],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Speed Trap (km/h)",
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Lap Time (s)",
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        autorange="reversed",
        range=[
            best_laps["lap_duration"].max() + 1.0,
            best_laps["lap_duration"].min() - 0.4,
        ],
        row=1,
        col=1,
    )

    leader = best_laps.iloc[0]
    leader_label = leader["name_acronym"] if "name_acronym" in best_laps.columns else leader["full_name"]
    winner_name = race_winner.get("full_name", "N/A") if is_race and race_winner else None
    winner_team = race_winner.get("team_name", "N/A") if is_race and race_winner else None

    fig.add_annotation(
        x=leader["st_speed_plot"],
        y=leader["lap_duration_plot"],
        xref="x1",
        yref="y1",
        text=(
            f"<b>Fastest Lap • {leader_label}</b><br>{leader['lap_duration']:.3f}s"
            if is_race
            else f"<b>P1 • {leader_label}</b><br>{leader['lap_duration']:.3f}s"
        ),
        showarrow=True,
        arrowhead=2,
        ax=45,
        ay=-35,
        bgcolor="rgba(15,23,42,0.90)",
        bordercolor="white",
        borderwidth=1,
        font=dict(size=11, color="white"),
    )

    top10 = best_laps.head(10).copy()
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Pos", "Piloto", "Equipe", "Lap", "Gap", "Vmax"] if not is_race else ["Pos FL", "Piloto", "Equipe", "Lap", "Gap", "Vmax"],
                fill_color="#1F2937",
                font=dict(color="white", size=12),
                align="left",
                height=30,
            ),
            cells=dict(
                values=[
                    top10["position"],
                    top10.get("full_name", pd.Series(["N/A"] * len(top10))),
                    top10.get("team_name", pd.Series(["N/A"] * len(top10))),
                    top10["lap_duration"].map(lambda x: f"{x:.3f}"),
                    top10["gap_to_leader"].map(lambda x: f"{x:.3f}"),
                    top10.get("st_speed", pd.Series([np.nan] * len(top10))).map(
                        lambda x: f"{x:.0f}" if pd.notna(x) else "-"
                    ),
                ],
                fill_color="#0F172A",
                font=dict(color="white", size=11),
                align="left",
                height=28,
            ),
        ),
        row=1,
        col=2,
    )

    x_gap = best_laps["name_acronym"] if "name_acronym" in best_laps.columns else best_laps["full_name"]
    fig.add_trace(
        go.Bar(
            x=x_gap,
            y=best_laps["gap_to_leader"],
            text=best_laps["gap_to_leader"].map(lambda x: f"{x:.3f}"),
            textposition="auto",
            textfont=dict(size=9),
            marker=dict(color=best_laps["team_color"]),
            hovertemplate="<b>%{x}</b><br>Gap: %{y:.3f}s<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(tickangle=-35, row=2, col=1)
    fig.update_yaxes(title_text="Gap (s)", showgrid=True, gridcolor=grid, zeroline=False, row=2, col=1)

    sector_colors = {"S1": "#F59E0B", "S2": "#EF4444", "S3": "#3B82F6"}

    for sector_name in ["S1", "S2", "S3"]:
        if sector_name not in rankings or rankings[sector_name].empty:
            continue

        df = rankings[sector_name].head(5).copy()
        labels = df["name_acronym"] if "name_acronym" in df.columns else df["full_name"]

        fig.add_trace(
            go.Bar(
                x=[f"{sector_name}-{x}" for x in labels],
                y=df["gap"],
                text=df["gap"].map(lambda x: f"{x:.3f}"),
                textposition="auto",
                textfont=dict(size=9),
                marker=dict(color=sector_colors[sector_name]),
                name=sector_name,
                hovertemplate="<b>%{x}</b><br>Gap: %{y:.3f}s<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig.update_yaxes(title_text="Gap setorial (s)", showgrid=True, gridcolor=grid, zeroline=False, row=2, col=2)

    if not team_summary.empty:
        fig.add_trace(
            go.Bar(
                x=team_summary["team_name"],
                y=team_summary["gap_to_best_team"],
                text=team_summary["gap_to_best_team"].map(lambda x: f"{x:.3f}"),
                textposition="auto",
                textfont=dict(size=9),
                marker=dict(color="#22C55E"),
                hovertemplate="<b>%{x}</b><br>Gap: %{y:.3f}s<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        fig.update_xaxes(tickangle=-25, row=3, col=1)
        fig.update_yaxes(
            title_text="Gap da melhor equipe (s)",
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            row=3,
            col=1,
        )

    has_sector_map = all(col in best_laps.columns for col in ["duration_sector_1", "duration_sector_2", "duration_sector_3"])
    if has_sector_map:
        customdata_map = np.column_stack([
            best_laps.get("full_name", pd.Series(["N/A"] * len(best_laps))).astype(str),
            best_laps.get("team_name", pd.Series(["N/A"] * len(best_laps))).astype(str),
            (
                best_laps["name_acronym"].astype(str)
                if "name_acronym" in best_laps.columns
                else best_laps.get("full_name", pd.Series(["N/A"] * len(best_laps))).astype(str)
            ),
            best_laps["duration_sector_1"].round(3).astype(str),
            best_laps["duration_sector_2"].round(3).astype(str),
            best_laps["duration_sector_3"].round(3).astype(str),
        ])

        fig.add_trace(
            go.Scatter(
                x=best_laps["duration_sector_1"],
                y=best_laps["duration_sector_2"],
                mode="markers",
                marker=dict(
                    size=15,
                    color=best_laps["team_color"],
                    line=dict(color="white", width=1),
                ),
                customdata=customdata_map,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Sigla: %{customdata[2]}<br>"
                    "Equipe: %{customdata[1]}<br>"
                    "S1: %{customdata[3]}s<br>"
                    "S2: %{customdata[4]}s<br>"
                    "S3: %{customdata[5]}s<extra></extra>"
                ),
                showlegend=False,
            ),
            row=3,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=top5["duration_sector_1"],
                y=top5["duration_sector_2"],
                mode="text",
                text=top5["name_acronym"] if "name_acronym" in top5.columns else top5["full_name"],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=2,
        )

        fig.update_xaxes(title_text="Sector 1 (s)", showgrid=True, gridcolor=grid, zeroline=False, row=3, col=2)
        fig.update_yaxes(title_text="Sector 2 (s)", showgrid=True, gridcolor=grid, zeroline=False, row=3, col=2)

    session_title = f"{session.get('country_name', '')} GP {session.get('year', '')} • {session.get('session_name', '')}"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{session_title}</b><br>"
                f"<span style='font-size:14px;color:{sub_font}'>Performance Dashboard • OpenF1 Data</span>"
            ),
            x=0.02,
            xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=panel,
        font=dict(color=font_color, family="Arial"),
        height=1480,
        margin=dict(l=40, r=40, t=165, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    fastest_speed_val = best_laps["st_speed"].max() if "st_speed" in best_laps.columns else None
    p1_team = leader.get("team_name", "N/A")
    p1_time = leader["lap_duration"]

    fig.add_annotation(
        x=0.02,
        y=1.13,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Vencedor:</b> {winner_name}"
            if is_race and winner_name
            else f"<b>P1:</b> {leader['full_name']} • {p1_time:.3f}s"
        ),
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=8,
    )

    fig.add_annotation(
        x=0.28,
        y=1.13,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Equipe vencedora:</b> {winner_team}"
            if is_race and winner_team
            else f"<b>Equipe líder:</b> {p1_team}"
        ),
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=8,
    )

    if fastest_speed_val is not None and pd.notna(fastest_speed_val):
        fig.add_annotation(
            x=0.53,
            y=1.13,
            xref="paper",
            yref="paper",
            text=f"<b>Maior Vmax:</b> {fastest_speed_val:.0f} km/h",
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            borderpad=8,
        )

    insights = build_insights(best_laps, rankings)
    insight_html = "<br>".join([f"• {x}" for x in insights[:5]])

    fig.add_annotation(
        x=0.985,
        y=1.06,
        xref="paper",
        yref="paper",
        text=f"<b>Insights</b><br>{insight_html}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="left",
        font=dict(size=12, color="white"),
        bgcolor="rgba(15,23,42,0.92)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=10,
    )

    return fig


def inject_brand_style():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.10), transparent 55%),
                radial-gradient(900px 600px at 100% 0%, rgba(16,185,129,0.10), transparent 50%),
                linear-gradient(180deg, #050A14 0%, #091224 100%);
        }
        .hero {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 22px 24px;
            background: linear-gradient(120deg, rgba(15,23,42,0.92), rgba(2,132,199,0.24));
            margin-bottom: 14px;
        }
        .hero h1 {
            margin: 0 0 4px 0;
            color: #E2E8F0;
            font-size: 30px;
            letter-spacing: 0.2px;
        }
        .hero p {
            margin: 0;
            color: #BFDBFE;
            font-size: 14px;
        }
        .pill-row {
            margin-top: 12px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .pill {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 999px;
            padding: 4px 10px;
            color: #DBEAFE;
            font-size: 12px;
            background: rgba(15, 23, 42, 0.6);
        }
        .kpi-card {
            border: 1px solid rgba(100, 116, 139, 0.35);
            border-radius: 14px;
            padding: 12px 14px;
            background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(15,23,42,0.55));
            min-height: 90px;
        }
        .kpi-label {
            color: #93C5FD;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #F8FAFC;
            font-size: 24px;
            font-weight: 700;
            line-height: 1.1;
        }
        .kpi-sub {
            margin-top: 4px;
            color: #94A3B8;
            font-size: 12px;
        }
        .insight-card {
            border: 1px solid rgba(51, 65, 85, 0.7);
            border-radius: 14px;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.72);
            min-height: 88px;
        }
        .insight-title {
            color: #67E8F9;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 6px;
        }
        .insight-text {
            color: #E2E8F0;
            font-size: 14px;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(session):
    session_title = (
        f"{session.get('country_name', 'N/A')} GP {session.get('year', 'N/A')} • "
        f"{session.get('session_name', 'N/A')}"
    )
    meeting = session.get("meeting_name", "OpenF1")
    date_start = pd.to_datetime(session.get("date_start"), errors="coerce", utc=True)
    date_label = date_start.strftime("%d/%m/%Y") if pd.notna(date_start) else "Data indisponível"
    st.markdown(
        f"""
        <div class="hero">
            <h1>F1 Intelligence Studio</h1>
            <p>{session_title} | {meeting} | {date_label}</p>
            <div class="pill-row">
                <span class="pill">Live Analytics</span>
                <span class="pill">Session Storytelling</span>
                <span class="pill">Commercial-ready Dashboard</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, sub):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(title, text):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_executive_insights(best_laps, rankings, team_summary, laps_enriched, session_type=None, race_winner=None):
    if best_laps.empty:
        return []

    insights = []
    leader = best_laps.iloc[0]
    if session_type == "race" and race_winner is not None:
        winner_name = race_winner.get("full_name", "N/A")
        winner_team = race_winner.get("team_name", "N/A")
        insights.append(("Headline", f"Vitória oficial: {winner_name} ({winner_team}) no resultado final da corrida."))
        insights.append(("Melhor volta", f"Fastest lap da sessão: {leader['full_name']} com {leader['lap_duration']:.3f}s."))
    else:
        p2_gap = best_laps.iloc[1]["gap_to_leader"] if len(best_laps) > 1 else 0.0
        insights.append(("Headline", f"{leader['full_name']} dominou a sessão com margem de {p2_gap:.3f}s para o P2."))

    if "st_speed" in best_laps.columns and best_laps["st_speed"].notna().any():
        v_row = best_laps.loc[best_laps["st_speed"].idxmax()]
        if v_row["full_name"] != leader["full_name"]:
            insights.append(
                (
                    "Eficiência de volta",
                    f"Maior Vmax ficou com {v_row['full_name']}; liderança no tempo veio de eficiência nos setores.",
                )
            )

    if session_type == "race" and race_winner is not None:
        insights.append(("Equipe vencedora", f"{race_winner.get('team_name', 'N/A')} venceu a corrida no resultado oficial."))
    elif not team_summary.empty and len(team_summary) > 1:
        t1 = team_summary.iloc[0]
        t2_gap = team_summary.iloc[1]["gap_to_best_team"]
        insights.append(("Força de equipe", f"{t1['team_name']} lidera o grid com {t2_gap:.3f}s sobre a 2ª equipe."))

    if "S1" in rankings and "S2" in rankings and "S3" in rankings:
        s1 = rankings["S1"].iloc[0]["team_name"] if not rankings["S1"].empty else "N/A"
        s2 = rankings["S2"].iloc[0]["team_name"] if not rankings["S2"].empty else "N/A"
        s3 = rankings["S3"].iloc[0]["team_name"] if not rankings["S3"].empty else "N/A"
        insights.append(("Leitura setorial", f"Referências por setor: S1={s1}, S2={s2}, S3={s3}."))

    if not laps_enriched.empty:
        valid = laps_enriched["lap_duration"].dropna()
        if len(valid) >= 10:
            spread = valid.quantile(0.9) - valid.quantile(0.1)
            insights.append(("Volatilidade", f"Dispersão operacional da sessão: {spread:.3f}s entre p10 e p90 das voltas."))

    return insights[:4]


def main():
    st.set_page_config(page_title="F1 OpenF1 Dashboard", layout="wide")
    inject_brand_style()
    initialize_session_state()

    with st.sidebar:
        st.header("Configuração")
        year = st.number_input("Ano", min_value=2018, max_value=2035, value=APP_CONFIG.default_year, step=1)
        country_filter = st.text_input("Filtrar país", value=APP_CONFIG.default_country_filter)

    sessions = filter_sessions_by_country(get_sessions(int(year)), country_filter)
    if sessions.empty:
        st.warning("Nenhuma sessão encontrada para os filtros informados.")
        return

    session_options = sessions.to_dict("records")

    with st.sidebar:
        selected_session = st.selectbox(
            "Sessão disponível",
            options=session_options,
            format_func=format_session_option,
            index=max(len(session_options) - 1, 0),
        )
        run = st.button("Carregar sessão", type="primary", use_container_width=True)

    if run:
        st.session_state["selected_session"] = selected_session
        st.session_state["run_dashboard"] = True

    if not st.session_state["run_dashboard"] or st.session_state["selected_session"] is None:
        st.markdown(
            """
            <div class="hero">
                <h1>F1 Intelligence Studio</h1>
                <p>Configure a sessão na barra lateral e clique em carregar para abrir um dashboard comercial com insights automáticos.</p>
                <div class="pill-row">
                    <span class="pill">Análise de performance</span>
                    <span class="pill">Estratégia de pneus</span>
                    <span class="pill">Traçado e delta</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Consultando OpenF1 e montando dashboard..."):
        try:
            session = st.session_state["selected_session"]
            session_key = session["session_key"]
            session_type = normalize_session_type(session)

            data = fetch_session_data(session_key, session_type)
            best_laps = compute_best_laps(data["laps"], data["drivers"])
            rankings = sector_rankings(best_laps)
            team_summary = compute_team_summary(best_laps)
            lap_times = prepare_lap_times(data["laps"], data["drivers"])
            laps_enriched = enrich_laps_with_stints(lap_times, data["stints"])
            laps_classified = classify_laps_advanced(laps_enriched)
            long_run_df = compute_long_run_summary(laps_classified)
            driver_score, team_score = build_scorecards(best_laps, laps_classified, long_run_df)
            driver_score_fig, team_score_fig = create_scorecard_charts(driver_score, team_score)
            teammate_metrics = prepare_teammate_metrics(best_laps, laps_classified)
            race_winner = get_race_winner(data["session_result"], data["drivers"]) if session_type == "race" else None
            fig = create_professional_infographic(
                best_laps, rankings, team_summary, session, session_type=session_type, race_winner=race_winner
            )

        except Exception as exc:
            st.error(f"Falha ao gerar dashboard: {exc}")
            return

    render_hero(session)

    leader = best_laps.iloc[0]
    vmax_value = f"{best_laps['st_speed'].max():.0f} km/h" if "st_speed" in best_laps.columns and best_laps["st_speed"].notna().any() else "N/A"
    teams_count = best_laps["team_name"].nunique() if "team_name" in best_laps.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if session_type == "race" and race_winner is not None:
            render_kpi_card("Vencedor", race_winner.get("full_name", "N/A"), f"Equipe: {race_winner.get('team_name', 'N/A')}")
        else:
            render_kpi_card("P1 da Sessão", leader.get("full_name", "N/A"), f"Tempo: {leader['lap_duration']:.3f}s")
    with k2:
        render_kpi_card("Melhor volta", f"{leader['lap_duration']:.3f}s", f"Piloto: {leader.get('full_name', 'N/A')}")
    with k3:
        render_kpi_card("Maior Vmax", vmax_value, "Velocidade máxima registrada")
    with k4:
        render_kpi_card("Equipes no Grid", f"{teams_count}", "Cobertura competitiva da sessão")

    exec_insights = build_executive_insights(
        best_laps, rankings, team_summary, laps_enriched, session_type=session_type, race_winner=race_winner
    )
    if exec_insights:
        i_cols = st.columns(len(exec_insights))
        for idx, (title, text) in enumerate(exec_insights):
            with i_cols[idx]:
                render_insight_card(title, text)

    st.markdown("")

    if session_type == "race":
        tab_exec, tab_race, tab_pilots, tab_sectors, tab_strategy, tab_data = st.tabs(
            ["Visão Executiva", "Corrida", "Pilotos", "Setores", "Estratégia", "Dados & Conteúdo"]
        )
    else:
        tab_exec, tab_pilots, tab_sectors, tab_strategy, tab_data = st.tabs(
            ["Visão Executiva", "Pilotos", "Setores", "Estratégia", "Dados & Conteúdo"]
        )
        tab_race = None

    comp_source, driver_options, driver_label_map, default_drivers = build_driver_selection(best_laps)

    selected_drivers = st.multiselect(
        "Pilotos foco (aplica em Pilotos/Setores/Estratégia)",
        options=driver_options,
        default=default_drivers,
        format_func=lambda d: driver_label_map.get(d, str(d)),
    )
    with tab_exec:
        show_chart(fig)
        csc1, csc2 = st.columns(2)
        with csc1:
            if driver_score_fig is not None:
                show_chart(driver_score_fig)
        with csc2:
            if team_score_fig is not None:
                show_chart(team_score_fig)
        if not driver_score.empty:
            st.subheader("Scorecard por piloto")
            cols_score = [
                c
                for c in [
                    "full_name",
                    "team_name",
                    "overall_score",
                    "pace_score",
                    "consistency_score",
                    "degradation_score",
                    "execution_score",
                ]
                if c in driver_score.columns
            ]
            st.dataframe(
                driver_score[cols_score].head(20).round(2),
                use_container_width=True,
                hide_index=True,
            )
        if session_type == "race" and not data["session_result"].empty:
            st.subheader("Classificação oficial da corrida")
            result_df = data["session_result"].copy()
            pos_col = next((c for c in ["position", "classification_position", "rank"] if c in result_df.columns), None)
            if pos_col is not None:
                result_df[pos_col] = pd.to_numeric(result_df[pos_col], errors="coerce")
                result_df = result_df.dropna(subset=[pos_col]).sort_values(pos_col)
            dcols = [c for c in ["driver_number", "full_name", "name_acronym", "team_name"] if c in data["drivers"].columns]
            if dcols and "driver_number" in result_df.columns:
                dmeta = data["drivers"][dcols].drop_duplicates(subset=["driver_number"]).copy()
                dmeta["driver_number"] = pd.to_numeric(dmeta["driver_number"], errors="coerce")
                result_df["driver_number"] = pd.to_numeric(result_df["driver_number"], errors="coerce")
                result_df = result_df.merge(dmeta, on="driver_number", how="left", suffixes=("", "_drv"))
                if "full_name" not in result_df.columns:
                    result_df["full_name"] = np.nan
                if "team_name" not in result_df.columns:
                    result_df["team_name"] = np.nan
                if "full_name_drv" in result_df.columns:
                    result_df["full_name"] = result_df["full_name"].fillna(result_df["full_name_drv"])
                if "team_name_drv" in result_df.columns:
                    result_df["team_name"] = result_df["team_name"].fillna(result_df["team_name_drv"])
            show_result_cols = [c for c in [pos_col, "full_name", "team_name", "points"] if c in result_df.columns]
            if show_result_cols:
                st.dataframe(result_df[show_result_cols].head(20), use_container_width=True, hide_index=True)
        else:
            show_cols = [
                col
                for col in ["position", "full_name", "team_name", "lap_duration", "gap_to_leader", "st_speed"]
                if col in best_laps.columns
            ]
            st.subheader("Top 10 best laps")
            st.dataframe(best_laps[show_cols].head(10), use_container_width=True, hide_index=True)
        st.subheader("Insights automáticos")
        for item in build_insights(best_laps, rankings):
            st.markdown(f"- {item}")

    if tab_race is not None:
        with tab_race:
            st.subheader("Evolução de posição por volta")
            pos_drivers = st.multiselect(
                "Pilotos para evolução de posição",
                options=driver_options,
                default=driver_options[: min(8, len(driver_options))],
                format_func=lambda d: driver_label_map.get(d, str(d)),
                key="race_pos_drivers",
            )
            race_pos_fig = create_race_position_evolution_chart(
                data.get("position", pd.DataFrame()),
                data.get("laps", pd.DataFrame()),
                data.get("drivers", pd.DataFrame()),
                pos_drivers if pos_drivers else driver_options[: min(8, len(driver_options))],
                only_finishers=False,
                completion_ratio=0.9,
            )
            show_chart_or_info(race_pos_fig, "Sem dados suficientes para evolução de posição por volta.")

            st.subheader("Linha do tempo da corrida")
            timeline_fig = create_session_timeline(
                laps_enriched,
                data["stints"],
                data["race_control"],
                data["weather"],
                selected_drivers if selected_drivers else default_drivers,
            )
            show_chart_or_info(timeline_fig, "Sem dados suficientes para linha do tempo da corrida.")

    with tab_pilots:
        st.subheader("Comparativo entre pilotos")
        compare_drivers = st.multiselect(
            "Selecione 2 a 4 pilotos para o comparativo",
            options=driver_options,
            default=driver_options[: min(4, len(driver_options))],
            format_func=lambda d: driver_label_map.get(d, str(d)),
            key="compare_section_drivers",
            max_selections=4,
        )

        if len(compare_drivers) < 2:
            st.info("Selecione ao menos 2 pilotos para habilitar os comparativos.")
        else:
            compare_df = best_laps[best_laps["driver_number"].isin(compare_drivers)].copy()
            cols = [c for c in ["position", "full_name", "team_name", "lap_duration", "gap_to_leader", "st_speed"] if c in compare_df.columns]
            st.dataframe(compare_df[cols].sort_values("position"), use_container_width=True, hide_index=True)

            compare_bar = create_driver_comparison_chart(best_laps, compare_drivers)
            show_chart_or_info(compare_bar, "Sem dados para comparação direta dos pilotos selecionados.")

            c_left, c_right = st.columns(2)
            with c_left:
                evo_fig = create_lap_evolution_chart(lap_times, compare_drivers)
                show_chart_or_info(evo_fig, "Sem dados para evolução de ritmo dos pilotos selecionados.")
            with c_right:
                consistency_fig = create_consistency_boxplot(lap_times, compare_drivers)
                show_chart_or_info(consistency_fig, "Sem dados para consistência dos pilotos selecionados.")

            st.subheader("Delta de telemetria entre 2 pilotos")
            delta_default_compare = compare_drivers[:2]
            selected_delta_drivers = st.multiselect(
                "Selecione exatamente 2 pilotos para delta",
                options=compare_drivers,
                default=delta_default_compare,
                format_func=lambda d: driver_label_map.get(d, str(d)),
                max_selections=2,
                key="delta_drivers_compare",
            )
            if len(selected_delta_drivers) != 2:
                st.info("Selecione exatamente 2 pilotos para gerar o delta.")
            else:
                d1, d2 = selected_delta_drivers[0], selected_delta_drivers[1]
                trace_1 = build_driver_delta_trace(
                    data["laps"], data["location"], data.get("position", pd.DataFrame()), session_key, d1
                )
                trace_2 = build_driver_delta_trace(
                    data["laps"], data["location"], data.get("position", pd.DataFrame()), session_key, d2
                )
                delta_fig = create_telemetry_delta_chart(
                    trace_1,
                    trace_2,
                    driver_label_map.get(d1, str(d1)),
                    driver_label_map.get(d2, str(d2)),
                )
                show_chart_or_info(
                    delta_fig,
                    "Não foi possível montar o delta para esses pilotos nessa sessão "
                    "(faltam dados de localização e/ou car_data na melhor volta).",
                )

        st.subheader("Teammate Intelligence")
        if teammate_metrics.empty or "team_name" not in teammate_metrics.columns:
            st.info("Sem dados suficientes para painel teammate.")
        else:
            team_counts = teammate_metrics.groupby("team_name")["driver_number"].nunique().reset_index(name="n")
            valid_teams = team_counts[team_counts["n"] >= 2]["team_name"].tolist()
            if not valid_teams:
                st.info("Nenhuma equipe com ao menos 2 pilotos válidos nesta sessão.")
            else:
                selected_team = st.selectbox("Equipe para duelo interno", options=valid_teams, key="teammate_team")
                team_df = teammate_metrics[teammate_metrics["team_name"] == selected_team].copy()
                team_df = team_df.sort_values("lap_duration").head(2).reset_index(drop=True)
                show_tm_cols = [
                    c
                    for c in [
                        "full_name",
                        "name_acronym",
                        "lap_duration",
                        "duration_sector_1",
                        "duration_sector_2",
                        "duration_sector_3",
                        "consistency_std",
                        "push_ratio",
                    ]
                    if c in team_df.columns
                ]
                team_show = team_df[show_tm_cols].copy()
                if "push_ratio" in team_show.columns:
                    team_show["push_ratio"] = (team_show["push_ratio"] * 100).round(1)
                st.dataframe(team_show.round(3), use_container_width=True, hide_index=True)

                tm_left, tm_right = st.columns(2)
                with tm_left:
                    tm_timing = create_teammate_timing_chart(team_df)
                    if tm_timing is not None:
                        show_chart(tm_timing)
                with tm_right:
                    tm_ops = create_teammate_ops_chart(team_df)
                    if tm_ops is not None:
                        show_chart(tm_ops)

                for line in build_teammate_summary(team_df):
                    st.markdown(f"- {line}")

    with tab_sectors:
        st.subheader("Heatmap setorial")
        heatmap_fig = create_sector_delta_heatmap(best_laps)
        show_chart_or_info(heatmap_fig, "Sem dados setoriais para heatmap.")

        st.subheader("Setores em 3D")
        sector_3d_fig = create_sector_3d_chart(best_laps)
        show_chart_or_info(sector_3d_fig, "Sem dados setoriais suficientes para visual 3D.")

    with tab_strategy:
        st.subheader("Tyre analytics")
        pace_fig, degr_fig, compare_fig = create_tyre_analytics_charts(laps_enriched)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            show_chart_or_info(pace_fig, "Sem dados para pace por composto.")
        with col_t2:
            show_chart_or_info(compare_fig, "Sem dados para comparação inlap/outlap vs push.")
        show_chart_or_info(degr_fig, "Sem dados suficientes para degradação por stint.")

        st.subheader("Classificação automática de voltas")
        phase_fig = create_lap_phase_distribution_chart(
            laps_classified,
            selected_drivers if selected_drivers else default_drivers,
        )
        show_chart_or_info(phase_fig, "Sem dados para classificação automática de voltas.")

        st.subheader("Ritmo e consistência (pilotos foco)")
        evo_fig = create_lap_evolution_chart(lap_times, selected_drivers if selected_drivers else default_drivers)
        show_chart_or_info(evo_fig, "Sem dados suficientes para evolução temporal.")
        consistency_fig = create_consistency_boxplot(lap_times, selected_drivers if selected_drivers else default_drivers)
        show_chart_or_info(consistency_fig, "Sem dados suficientes para análise de consistência.")

        st.subheader("Long-run analytics")
        long_run_fig = create_long_run_chart(long_run_df)
        if long_run_fig is not None:
            show_chart(long_run_fig)
            lr_cols = [
                c
                for c in [
                    "driver_label",
                    "team_name",
                    "compound",
                    "stint_number",
                    "laps",
                    "avg_pace",
                    "consistency_std",
                    "degradation_s_per_lap",
                ]
                if c in long_run_df.columns
            ]
            st.dataframe(long_run_df[lr_cols].round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Sem volume de voltas push suficiente para long-run por stint.")

    with tab_data:
        st.subheader("Texto pronto para post")
        st.code(
            build_marketing_text(
                best_laps,
                rankings,
                session.get("country_name", "N/A"),
                session.get("year", int(year)),
                session.get("session_name", "N/A"),
            ),
            language="text",
        )

        with st.expander("Dados brutos"):
            for name, df in data.items():
                st.write(f"**{name}** — {df.shape[0]} linhas")
                st.dataframe(df.head(30), use_container_width=True)


if __name__ == "__main__":
    main()
