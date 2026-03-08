"""Reusable Streamlit UI blocks and branding styles for the dashboard."""

import pandas as pd
import streamlit as st


def inject_brand_style():
    """Inject custom CSS to define the visual identity of the app."""

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
    """Render the top hero banner with session context metadata."""

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
    """Render a compact KPI card with label, value, and subtext."""

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
    """Render a compact executive-insight card."""

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
