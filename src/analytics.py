"""Data transformations and analytical features for dashboard insights."""

import pandas as pd
import numpy as np


def compute_best_laps(laps, drivers):
    """Compute each driver's best lap and ranking relative to session leader."""

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
    """Build per-sector ranking tables and gaps to the fastest sector lap."""

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
    """Aggregate team-level pace and speed indicators from best laps."""

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

def _to_bool_flag(series):
    """Normalize heterogeneous boolean-like values into a bool Series."""

    if series is None:
        return pd.Series(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["1", "true", "t", "yes", "y"])

def classify_laps_advanced(laps_enriched):
    """Classify laps into push/traffic/inlap/outlap phases using robust heuristics."""

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

def compute_long_run_summary(laps_classified):
    """Summarize long-run pace, consistency, and degradation by driver stint."""

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

def build_scorecards(best_laps, laps_classified, long_run_df):
    """Create weighted executive scorecards for drivers and teams."""

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

def prepare_teammate_metrics(best_laps, laps_classified):
    """Prepare teammate-comparison dataset with timing and execution metrics."""

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

def build_teammate_summary(team_df):
    """Generate textual bullets comparing two teammates."""

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

def build_insights(best_laps, rankings):
    """Generate high-level session insights from pace, speed, and sectors."""

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
    """Build ready-to-post marketing copy for social content."""

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

def build_executive_insights(best_laps, rankings, team_summary, laps_enriched, session_type=None, race_winner=None):
    """Generate executive cards focused on outcomes and decision context."""

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
