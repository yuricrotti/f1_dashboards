"""Plotly chart builders used across dashboard sections."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.helpers import safe_team_color
from src.analytics import build_insights
from src.data_layer import derive_stint_windows, fetch_driver_car_data


def create_session_timeline(laps_enriched, stints, race_control, weather, selected_driver_numbers):
    """Create a unified timeline view with laps, stints, race control, and weather."""

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
    """Build tyre-related charts: compound pace, stint degradation, and lap-type comparison."""

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

def create_lap_phase_distribution_chart(laps_classified, selected_driver_numbers):
    """Create distribution chart for classified lap phases."""

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

def create_long_run_chart(long_run_df):
    """Plot long-run tradeoff between degradation and average pace."""

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

def create_scorecard_charts(driver_score, team_score):
    """Create bar charts for driver and team executive scores."""

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

def create_teammate_timing_chart(team_df):
    """Compare teammate lap and sector times in grouped bars."""

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
    """Compare teammate consistency and execution in a dual-axis chart."""

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

def build_driver_delta_trace(laps, location, position, session_key, driver_number):
    """Build normalized lap trace (progress vs elapsed time) for telemetry delta."""

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
    """Create cumulative delta chart between two normalized traces."""

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
    """Normalize telemetry coordinates into `x`/`y` columns."""

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
    """Create direct comparison chart for selected drivers and sector splits."""

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
    """Create lap-by-lap pace evolution chart for selected drivers."""

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
    """Create lap-time distribution boxplots for consistency analysis."""

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
    """Build race position progression by lap, with optional finisher filtering."""

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
    """Create heatmap of sector deltas versus session leader."""

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
    """Create 3D sector-space scatter (S1, S2, S3) by driver."""

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

def create_professional_infographic(best_laps, rankings, team_summary, session, session_type=None, race_winner=None):
    """Create the flagship multi-panel infographic dashboard figure."""

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
