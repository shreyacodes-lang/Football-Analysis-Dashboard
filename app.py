"""
app.py
──────
Football Player Performance Analytics Dashboard
Synthetic data — not real football statistics.

Run:
    streamlit run app.py
"""

import io
import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = pathlib.Path("football_players.csv")

ACCENT     = "#2ea043"   # pitch green
BG_DARK    = "#0d1117"
BG_CARD    = "#161b22"
TEXT_LIGHT = "#e6edf3"
TEXT_MUTED = "#8b949e"
BORDER     = "#30363d"

PLOTLY_TEMPLATE = "plotly_dark"

PAGES = ["Overview", "Player Analysis", "Team Analysis", "Position Analysis"]

POSITION_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

POSITION_COLORS = {
    "Goalkeeper": "#f0a500",
    "Defender":   "#2ea043",
    "Midfielder": "#3b82f6",
    "Forward":    "#ef4444",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ── App background ── */
        .stApp {{
            background-color: {BG_DARK};
            color: {TEXT_LIGHT};
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {BG_CARD};
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stSidebar"] * {{
            color: {TEXT_LIGHT} !important;
        }}

        /* ── KPI card ── */
        .kpi-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-top: 3px solid {ACCENT};
            border-radius: 8px;
            padding: 18px 16px 14px 16px;
            text-align: center;
            margin-bottom: 8px;
        }}
        .kpi-label {{
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 700;
            color: {ACCENT};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 11px;
            color: {TEXT_MUTED};
            margin-top: 4px;
        }}

        /* ── Section heading ── */
        .section-heading {{
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
            border-bottom: 1px solid {BORDER};
            padding-bottom: 6px;
            margin-bottom: 16px;
        }}

        /* ── Synthetic data banner ── */
        .synth-banner {{
            background: #1a1f2e;
            border: 1px solid #3b4a6b;
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 12px;
            color: #93aed4;
            margin-bottom: 18px;
        }}

        /* ── Streamlit element overrides ── */
        div[data-testid="stMetricValue"] {{
            color: {ACCENT};
        }}
        .stDataFrame {{
            border: 1px solid {BORDER};
            border-radius: 6px;
        }}
        h1, h2, h3 {{
            color: {TEXT_LIGHT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Read CSV, validate, cast types, and drop unusable rows."""
    if not CSV_PATH.exists():
        st.error(
            "**football_players.csv not found.**  \n"
            "Run `python generate_data.py` in the project folder first, then refresh this page."
        )
        st.stop()

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "player_id", "player_name", "season", "team", "position", "age",
        "matches_played", "minutes_played", "goals", "assists",
        "shots", "shots_on_target", "pass_accuracy",
        "tackles", "interceptions", "yellow_cards", "red_cards", "rating",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"CSV is missing expected columns: {missing_cols}")
        st.stop()

    before = len(df)
    df = df.dropna(subset=["goals", "assists", "rating"])
    dropped = before - len(df)

    # Type safety
    int_cols = [
        "goals", "assists", "shots", "shots_on_target",
        "tackles", "interceptions", "yellow_cards", "red_cards",
        "matches_played", "minutes_played", "age",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["pass_accuracy"] = pd.to_numeric(df["pass_accuracy"], errors="coerce").fillna(0.0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    # Categorical ordering for position
    df["position"] = pd.Categorical(df["position"], categories=POSITION_ORDER, ordered=True)

    return df, dropped


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & FILTERS
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> dict:
    """Render all sidebar filter widgets and return the current filter state."""
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center; padding: 10px 0 20px 0;'>"
            f"<span style='font-size:28px;'>&#9917;</span>"
            f"<div style='font-size:15px; font-weight:700; color:{ACCENT}; margin-top:4px;'>"
            f"Football Analytics</div>"
            f"<div style='font-size:11px; color:{TEXT_MUTED};'>Dashboard</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Filters**")

        seasons_all = sorted(df["season"].unique().tolist())
        selected_seasons = st.multiselect("Season", seasons_all, default=seasons_all)

        teams_pool = df[df["season"].isin(selected_seasons)]["team"].unique().tolist() if selected_seasons else df["team"].unique().tolist()
        teams_all = sorted(teams_pool)
        selected_teams = st.multiselect("Team", teams_all, default=teams_all)

        pos_pool = (
            df[df["team"].isin(selected_teams)]["position"].unique().tolist()
            if selected_teams else POSITION_ORDER
        )
        pos_all = [p for p in POSITION_ORDER if p in pos_pool]
        selected_positions = st.multiselect("Position", pos_all, default=pos_all)

        player_pool = (
            df[
                df["season"].isin(selected_seasons)
                & df["team"].isin(selected_teams)
                & df["position"].isin(selected_positions)
            ]["player_name"].unique().tolist()
            if (selected_seasons and selected_teams and selected_positions)
            else []
        )
        player_all = sorted(player_pool)
        selected_players = st.multiselect("Player", player_all, default=player_all)

        st.markdown("---")

    return {
        "seasons":   selected_seasons,
        "teams":     selected_teams,
        "positions": selected_positions,
        "players":   selected_players,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Return a filtered copy of df based on the sidebar filter state."""
    mask = pd.Series(True, index=df.index)

    if filters["seasons"]:
        mask &= df["season"].isin(filters["seasons"])
    if filters["teams"]:
        mask &= df["team"].isin(filters["teams"])
    if filters["positions"]:
        mask &= df["position"].isin(filters["positions"])
    if filters["players"]:
        mask &= df["player_name"].isin(filters["players"])

    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _chart_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark layout to any Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_LIGHT, size=12),
        title=dict(text=title, font=dict(size=14, color=TEXT_LIGHT), x=0.01),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            bgcolor=BG_DARK,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


def kpi_card(label: str, value: str, sub: str = "") -> str:
    """Return HTML string for a KPI card."""
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{sub_html}"
        f"</div>"
    )


def section_heading(text: str) -> None:
    st.markdown(f"<div class='section-heading'>{text}</div>", unsafe_allow_html=True)


def synth_banner() -> None:
    st.markdown(
        "<div class='synth-banner'>"
        "&#9432;&nbsp; <strong>Synthetic Dataset</strong> — all player names, clubs, and statistics "
        "are artificially generated for demonstration purposes only. "
        "These figures do not represent real football players or events."
        "</div>",
        unsafe_allow_html=True,
    )


def _empty_state(msg: str = "No players match the selected filters.") -> None:
    st.warning(msg)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame) -> None:
    synth_banner()
    section_heading("Key Performance Indicators")

    total_players   = df["player_id"].nunique()
    total_goals     = int(df["goals"].sum())
    total_assists   = int(df["assists"].sum())
    total_minutes   = int(df["minutes_played"].sum())
    avg_rating      = round(df["rating"].mean(), 2)

    cols = st.columns(5)
    cards = [
        ("Total Players",     f"{total_players:,}",   ""),
        ("Total Goals",       f"{total_goals:,}",      ""),
        ("Total Assists",     f"{total_assists:,}",    ""),
        ("Minutes Played",    f"{total_minutes:,}",   ""),
        ("Avg Player Rating", f"{avg_rating:.2f}",    "out of 9.5"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(kpi_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    # ── Top-10 scorers ──
    with col_left:
        section_heading("Top 10 Goal Scorers")
        top_scorers = (
            df.groupby("player_name", as_index=False)
            .agg(goals=("goals", "sum"), team=("team", "first"), position=("position", "first"))
            .nlargest(10, "goals")
            .sort_values("goals")
        )
        fig_top = px.bar(
            top_scorers,
            x="goals",
            y="player_name",
            orientation="h",
            color="position",
            color_discrete_map=POSITION_COLORS,
            hover_data={"team": True, "position": True},
            labels={"goals": "Goals", "player_name": "Player", "position": "Position"},
            text="goals",
        )
        fig_top.update_traces(textposition="outside")
        fig_top = _chart_layout(fig_top)
        fig_top.update_yaxes(tickfont=dict(size=11))
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Goals by position donut ──
    with col_right:
        section_heading("Goals by Position")
        pos_goals = df.groupby("position", as_index=False)["goals"].sum()
        fig_donut = px.pie(
            pos_goals,
            names="position",
            values="goals",
            hole=0.45,
            color="position",
            color_discrete_map=POSITION_COLORS,
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Goals: %{value}<br>Share: %{percent}",
        )
        fig_donut = _chart_layout(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PLAYER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_player_analysis(df: pd.DataFrame) -> None:
    synth_banner()

    # ── Player comparison ──
    section_heading("Player Comparison")
    all_players = sorted(df["player_name"].unique().tolist())

    default_compare = all_players[:5] if len(all_players) >= 5 else all_players
    selected = st.multiselect(
        "Select players to compare (2–5):",
        all_players,
        default=default_compare,
        key="compare_select",
    )

    if len(selected) < 2:
        st.info("Select at least 2 players to see the comparison chart.")
    else:
        compare_df = (
            df[df["player_name"].isin(selected)]
            .groupby("player_name", as_index=False)
            .agg(
                Goals=("goals", "sum"),
                Assists=("assists", "sum"),
                Shots=("shots", "sum"),
                Tackles=("tackles", "sum"),
                Rating=("rating", "mean"),
            )
        )
        compare_df["Rating"] = compare_df["Rating"].round(2)

        stats_cols = ["Goals", "Assists", "Shots", "Tackles", "Rating"]
        fig_compare = go.Figure()
        for _, row in compare_df.iterrows():
            fig_compare.add_trace(
                go.Bar(
                    name=row["player_name"],
                    x=stats_cols,
                    y=[row[s] for s in stats_cols],
                    hovertemplate=f"<b>{row['player_name']}</b><br>%{{x}}: %{{y}}<extra></extra>",
                )
            )
        fig_compare.update_layout(barmode="group")
        fig_compare = _chart_layout(fig_compare, "Player Stats Comparison")
        st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter: Goals vs Assists ──
    col_l, col_r = st.columns(2)

    with col_l:
        section_heading("Goals vs Assists")
        scatter_df = (
            df.groupby(["player_name", "position", "team"], as_index=False)
            .agg(goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes_played", "sum"))
        )
        fig_ga = px.scatter(
            scatter_df,
            x="assists",
            y="goals",
            color="position",
            size="minutes",
            size_max=22,
            hover_name="player_name",
            hover_data={"team": True, "position": True, "minutes": True},
            color_discrete_map=POSITION_COLORS,
            labels={"goals": "Goals", "assists": "Assists", "minutes": "Minutes Played", "position": "Position"},
        )
        fig_ga = _chart_layout(fig_ga, "Goals vs Assists (bubble = minutes played)")
        st.plotly_chart(fig_ga, use_container_width=True)

    # ── Scatter: Shots vs Goals ──
    with col_r:
        section_heading("Shots vs Goals")
        shots_df = (
            df.groupby(["player_name", "position", "team"], as_index=False)
            .agg(goals=("goals", "sum"), shots=("shots", "sum"))
        )
        shots_df = shots_df[shots_df["shots"] > 0]

        try:
            fig_sg = px.scatter(
                shots_df,
                x="shots",
                y="goals",
                color="position",
                hover_name="player_name",
                hover_data={"team": True, "position": True},
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override=ACCENT,
                color_discrete_map=POSITION_COLORS,
                labels={"goals": "Goals", "shots": "Shots", "position": "Position"},
            )
        except Exception:
            fig_sg = px.scatter(
                shots_df,
                x="shots",
                y="goals",
                color="position",
                hover_name="player_name",
                hover_data={"team": True, "position": True},
                color_discrete_map=POSITION_COLORS,
                labels={"goals": "Goals", "shots": "Shots", "position": "Position"},
            )
        fig_sg = _chart_layout(fig_sg, "Shots vs Goals (conversion efficiency)")
        st.plotly_chart(fig_sg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Searchable stats table ──
    section_heading("Player Statistics Table")
    search_query = st.text_input("Search by player name or team:", placeholder="e.g. Marcus, Riverside FC...")

    display_cols = [
        "player_name", "season", "team", "position", "age",
        "matches_played", "minutes_played", "goals", "assists",
        "shots", "shots_on_target", "pass_accuracy",
        "tackles", "interceptions", "yellow_cards", "red_cards", "rating",
    ]
    table_df = df[display_cols].copy()
    table_df.columns = [c.replace("_", " ").title() for c in display_cols]

    if search_query:
        sq = search_query.lower()
        mask = (
            table_df["Player Name"].str.lower().str.contains(sq, na=False)
            | table_df["Team"].str.lower().str.contains(sq, na=False)
        )
        table_df = table_df[mask]

    st.caption(f"Showing {len(table_df):,} records")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ── CSV download ──
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="football_players_filtered.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — TEAM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_team_analysis(df: pd.DataFrame) -> None:
    synth_banner()

    team_agg = (
        df.groupby("team", as_index=False)
        .agg(
            Goals=("goals", "sum"),
            Assists=("assists", "sum"),
            Players=("player_id", "nunique"),
            Avg_Rating=("rating", "mean"),
        )
        .sort_values("Goals", ascending=False)
    )
    team_agg["Avg Rating"] = team_agg["Avg_Rating"].round(2)

    # ── Goals & Assists grouped bar ──
    section_heading("Team Goals & Assists")
    fig_ga = go.Figure()
    fig_ga.add_trace(go.Bar(name="Goals",   x=team_agg["team"], y=team_agg["Goals"],   marker_color=ACCENT,    hovertemplate="<b>%{x}</b><br>Goals: %{y}<extra></extra>"))
    fig_ga.add_trace(go.Bar(name="Assists", x=team_agg["team"], y=team_agg["Assists"], marker_color="#3b82f6", hovertemplate="<b>%{x}</b><br>Assists: %{y}<extra></extra>"))
    fig_ga.update_layout(barmode="group")
    fig_ga = _chart_layout(fig_ga, "Goals & Assists by Team")
    st.plotly_chart(fig_ga, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Player count per team ──
    with col_l:
        section_heading("Squad Depth (Player Count)")
        fig_pc = px.bar(
            team_agg.sort_values("Players", ascending=False),
            x="team", y="Players",
            color="Players",
            color_continuous_scale=[[0, "#1a4a2e"], [1, ACCENT]],
            labels={"team": "Team", "Players": "Unique Players"},
            text="Players",
        )
        fig_pc.update_traces(textposition="outside")
        fig_pc.update_layout(coloraxis_showscale=False)
        fig_pc = _chart_layout(fig_pc, "Players per Team")
        st.plotly_chart(fig_pc, use_container_width=True)

    # ── Average rating per team ──
    with col_r:
        section_heading("Average Player Rating by Team")
        team_rating = team_agg.sort_values("Avg Rating", ascending=False)
        fig_rt = px.bar(
            team_rating,
            x="team", y="Avg Rating",
            color="Avg Rating",
            color_continuous_scale=[[0, "#1a4a2e"], [1, ACCENT]],
            text="Avg Rating",
            labels={"team": "Team", "Avg Rating": "Avg Rating"},
        )
        fig_rt.update_traces(textposition="outside", texttemplate="%{text:.2f}")
        fig_rt.update_layout(coloraxis_showscale=False)
        fig_rt = _chart_layout(fig_rt, "Average Rating per Team")
        st.plotly_chart(fig_rt, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — POSITION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_position_analysis(df: pd.DataFrame) -> None:
    synth_banner()

    # ── Box plots ──
    section_heading("Stat Distributions by Position")
    box_stats = [
        ("goals",        "Goals"),
        ("assists",      "Assists"),
        ("pass_accuracy","Pass Accuracy (%)"),
        ("tackles",      "Tackles"),
        ("rating",       "Player Rating"),
    ]

    col1, col2 = st.columns(2)
    for i, (col_name, label) in enumerate(box_stats):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            fig_box = px.box(
                df,
                x="position",
                y=col_name,
                color="position",
                color_discrete_map=POSITION_COLORS,
                category_orders={"position": POSITION_ORDER},
                points="outliers",
                labels={"position": "Position", col_name: label},
            )
            fig_box = _chart_layout(fig_box, f"{label} Distribution")
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar chart ──
    section_heading("Positional Archetype Radar")
    radar_stats = ["goals", "assists", "pass_accuracy", "tackles", "shots", "interceptions"]
    radar_labels = ["Goals", "Assists", "Pass Acc.", "Tackles", "Shots", "Interceptions"]

    pos_avg = df.groupby("position", as_index=False)[radar_stats].mean()

    # Normalise each stat 0–1 across positions
    norm_df = pos_avg.copy()
    for stat in radar_stats:
        col_max = norm_df[stat].max()
        col_min = norm_df[stat].min()
        rng = col_max - col_min
        norm_df[stat] = (norm_df[stat] - col_min) / rng if rng > 0 else 0.0

    fig_radar = go.Figure()
    for _, row in norm_df.iterrows():
        pos = row["position"]
        values = [row[s] for s in radar_stats]
        values.append(values[0])  # close the polygon
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=str(pos),
                line=dict(color=POSITION_COLORS.get(str(pos), ACCENT), width=2),
                fillcolor=POSITION_COLORS.get(str(pos), ACCENT),
                opacity=0.25,
                hovertemplate="<b>" + str(pos) + "</b><br>%{theta}: %{r:.2f}<extra></extra>",
            )
        )
    fig_radar.update_layout(
        polar=dict(
            bgcolor=BG_DARK,
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=BORDER),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        ),
        showlegend=True,
    )
    fig_radar = _chart_layout(fig_radar, "Normalised Average Stats per Position (0=min, 1=max)")
    st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Football Analytics Dashboard",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # Load data
    df_raw, dropped_rows = load_data()

    # Sidebar filters
    filters = render_sidebar(df_raw)

    # Apply filters
    df = apply_filters(df_raw, filters)

    # Sidebar meta info
    with st.sidebar:
        st.caption(f"{len(df):,} records match current filters")
        if dropped_rows > 0:
            st.caption(f"({dropped_rows} rows dropped — missing values)")

        st.markdown("---")

        # Download in sidebar too
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered CSV",
            data=csv_bytes,
            file_name="football_filtered.csv",
            mime="text/csv",
        )

    # Page navigation
    with st.sidebar:
        st.markdown("---")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    # Page header
    st.markdown(
        f"<h1 style='color:{ACCENT}; margin-bottom:4px;'>Football Player Performance Dashboard</h1>"
        f"<p style='color:{TEXT_MUTED}; font-size:13px; margin-top:0;'>{page}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Guard: empty filter result
    if df.empty:
        _empty_state()
        return

    # Route to page
    if page == "Overview":
        page_overview(df)
    elif page == "Player Analysis":
        page_player_analysis(df)
    elif page == "Team Analysis":
        page_team_analysis(df)
    elif page == "Position Analysis":
        page_position_analysis(df)


if __name__ == "__main__":
    main()
