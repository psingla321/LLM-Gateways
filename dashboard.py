"""
LLM Gateway Dashboard
=====================
Live team-budget & usage analytics.

Run:  streamlit run dashboard.py
"""

import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gateway.logger import UsageLogger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Gateway Dashboard",
    page_icon="⚡",
    layout="wide",
)

DB_PATH = os.getenv("GATEWAY_DB", "usage.db")

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_data():
    if not os.path.exists(DB_PATH):
        return None, None, None, None
    logger   = UsageLogger(DB_PATH)
    summary  = logger.summary_stats()
    teams    = logger.team_budgets()
    models   = logger.model_stats()
    recents  = logger.recent_requests(limit=30)
    return summary, teams, models, recents


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ LLM Gateway — Live Dashboard")
st.caption("Insurance Claims · Real-time cost, routing, cache & PII analytics")

col_refresh, _ = st.columns([1, 9])
with col_refresh:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()

summary, teams, models, recents = load_data()

if summary is None or summary["total_requests"] == 0:
    st.info(
        "No data yet.  Run **`python demo.py`** first to populate the database, "
        "then refresh this page."
    )
    st.stop()

# ── Top KPI cards ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Total Requests",    summary["total_requests"])
k2.metric("Total Cost (USD)",  f"${summary['total_cost_usd']:.5f}")
k3.metric("Avg Latency",       f"{int(summary['avg_latency_ms'])} ms")
k4.metric("Cache Hit Rate",    f"{summary['cache_hit_rate']} %")
k5.metric("PII Masked",        summary["pii_masked_count"])
k6.metric("Fallbacks Used",    summary["fallback_count"])

st.divider()

# ── Row 1: Team budget + Model distribution ───────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("💰 Budget by Team")
    if teams:
        df_teams = pd.DataFrame(teams)
        fig_budget = px.bar(
            df_teams.sort_values("total_cost"),
            x="total_cost",
            y="team",
            orientation="h",
            color="team",
            text=df_teams.sort_values("total_cost")["total_cost"].apply(lambda v: f"${v:.5f}"),
            labels={"total_cost": "Cost (USD)", "team": "Team"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_budget.update_traces(textposition="outside")
        fig_budget.update_layout(
            showlegend=False,
            margin=dict(l=10, r=60, t=10, b=10),
            height=280,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_budget, use_container_width=True)

with right:
    st.subheader("🤖 Model Usage")
    if models:
        df_models = pd.DataFrame(models)
        fig_models = px.pie(
            df_models,
            names="model",
            values="requests",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_models.update_traces(textinfo="percent+label", textposition="outside")
        fig_models.update_layout(
            showlegend=True,
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_models, use_container_width=True)

# ── Row 2: Cache performance + Latency by model ───────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("⚡ Cache Performance")
    hits   = summary["cache_hits"]
    misses = summary["total_requests"] - hits
    fig_cache = go.Figure(go.Pie(
        labels=["Cache Hit", "Cache Miss"],
        values=[hits, misses],
        hole=0.55,
        marker_colors=["#2ecc71", "#e74c3c"],
        textinfo="value+percent",
    ))
    fig_cache.add_annotation(
        text=f"{summary['cache_hit_rate']}%<br>hit rate",
        x=0.5, y=0.5, font_size=18, showarrow=False,
    )
    fig_cache.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cache, use_container_width=True)

with c2:
    st.subheader("⏱️ Avg Latency by Model")
    if models:
        df_lat = pd.DataFrame(models)
        fig_lat = px.bar(
            df_lat.sort_values("avg_latency_ms", ascending=False),
            x="model",
            y="avg_latency_ms",
            color="model",
            text="avg_latency_ms",
            labels={"avg_latency_ms": "Avg Latency (ms)", "model": ""},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_lat.update_traces(texttemplate="%{text:.0f} ms", textposition="outside")
        fig_lat.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_lat, use_container_width=True)

# ── Row 3: Full team table ────────────────────────────────────────────────────
st.subheader("📊 Team Summary Table")
if teams:
    df_teams_full = pd.DataFrame(teams)
    df_teams_full["total_cost"] = df_teams_full["total_cost"].apply(lambda v: f"${v:.6f}")
    df_teams_full["avg_latency_ms"] = df_teams_full["avg_latency_ms"].apply(
        lambda v: f"{int(v or 0)} ms"
    )
    df_teams_full.columns = [
        "Team", "Requests", "Total Cost", "Total Tokens",
        "Avg Latency", "Cache Hits", "Complex Claims",
    ]
    st.dataframe(df_teams_full, use_container_width=True, hide_index=True)

# ── Row 4: Recent requests log ────────────────────────────────────────────────
st.subheader("📋 Recent Requests")
if recents:
    df_rec = pd.DataFrame(recents)

    # Styling helpers
    def _style_task(val):
        color = "#e74c3c" if val == "complex" else "#2ecc71"
        return f"color: {color}; font-weight: bold"

    def _style_cache(val):
        return "color: #2ecc71; font-weight: bold" if val else ""

    def _style_fallback(val):
        return "color: #e74c3c; font-weight: bold" if val else ""

    df_rec["cost_usd"]      = df_rec["cost_usd"].apply(lambda v: f"${v:.6f}")
    df_rec["latency_ms"]    = df_rec["latency_ms"].apply(lambda v: f"{v} ms")
    df_rec["cache_hit"]     = df_rec["cache_hit"].apply(lambda v: "✓ HIT" if v else "miss")
    df_rec["fallback_used"] = df_rec["fallback_used"].apply(lambda v: "⚠ YES" if v else "no")
    df_rec["pii_masked"]    = df_rec["pii_masked"].apply(lambda v: "🔒 YES" if v else "no")

    df_rec.rename(columns={
        "request_id":   "ID",
        "timestamp":    "Time",
        "team":         "Team",
        "user_id":      "User",
        "task_type":    "Task",
        "model_used":   "Model",
        "fallback_used":"Fallback",
        "tokens_in":    "Tok In",
        "tokens_out":   "Tok Out",
        "cost_usd":     "Cost",
        "latency_ms":   "Latency",
        "cache_hit":    "Cache",
        "pii_masked":   "PII",
        "provider":     "Provider",
    }, inplace=True)

    st.dataframe(
        df_rec[["ID", "Time", "Team", "User", "Task", "Model",
                "Provider", "Fallback", "Tok In", "Tok Out",
                "Cost", "Latency", "Cache", "PII"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.divider()
st.caption("Auto-refreshes every 10 s  ·  Data source: usage.db  ·  LLM Gateway demo")

# Trigger rerun every 10 seconds
time.sleep(10)
st.rerun()
