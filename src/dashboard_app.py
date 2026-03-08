"""Streamlit application orchestration for the F1 Intelligence dashboard."""

import pandas as pd
import streamlit as st

from src.config import APP_CONFIG
from src.helpers import initialize_session_state, build_driver_selection, show_chart, show_chart_or_info
from src.data_layer import (
    get_sessions,
    filter_sessions_by_country,
    format_session_option,
    normalize_session_type,
    fetch_session_data,
    get_race_winner,
    prepare_lap_times,
    enrich_laps_with_stints,
)
from src.analytics import (
    compute_best_laps,
    sector_rankings,
    compute_team_summary,
    classify_laps_advanced,
    compute_long_run_summary,
    build_scorecards,
    prepare_teammate_metrics,
    build_teammate_summary,
    build_insights,
    build_executive_insights,
)
from src.charts import (
    create_scorecard_charts,
    create_professional_infographic,
    create_driver_comparison_chart,
    create_lap_evolution_chart,
    create_consistency_boxplot,
    build_driver_delta_trace,
    create_telemetry_delta_chart,
    create_sector_delta_heatmap,
    create_sector_3d_chart,
    create_race_position_evolution_chart,
    create_session_timeline,
    create_tyre_analytics_charts,
    create_lap_phase_distribution_chart,
    create_long_run_chart,
    create_teammate_timing_chart,
    create_teammate_ops_chart,
)
from src.ui_components import inject_brand_style, render_hero, render_kpi_card, render_insight_card


def main():
    """Run the full Streamlit workflow: load data, compute analytics, render sections."""

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
        tab_exec, tab_race, tab_pilots, tab_sectors, tab_strategy = st.tabs(
            ["Visão Executiva", "Corrida", "Pilotos", "Setores", "Estratégia"]
        )
    else:
        tab_exec, tab_pilots, tab_sectors, tab_strategy = st.tabs(
            ["Visão Executiva", "Pilotos", "Setores", "Estratégia"]
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

if __name__ == "__main__":
    main()
