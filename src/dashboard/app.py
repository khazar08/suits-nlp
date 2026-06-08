"""
Suits Power Network — Streamlit Dashboard
Run: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = _root = Path(__file__).parent.parent.parent
DATA_DIR = _root / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = _root / "processed"


CHAR_COLORS = {
    "Harvey Specter":   "#1565C0",
    "Mike Ross":        "#E65100",
    "Louis Litt":       "#2E7D32",
    "Donna Paulsen":    "#C62828",
    "Jessica Pearson":  "#6A1B9A",
    "Rachel Zane":      "#4E342E",
    "Katrina Bennett":  "#AD1457",
    "Alex Williams":    "#37474F",
    "Samantha Wheeler": "#F9A825",
}
DEFAULT_COLOR = "#78909C"

MAIN_CHARS = list(CHAR_COLORS.keys())


# Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suits Power Network",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  h1, h2, h3 { color: #e8e8e8; }
  .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
  .metric-card { background: #1a1d27; border-radius: 8px; padding: 12px 16px; }
</style>
""", unsafe_allow_html=True)


#  Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _warn(msg: str) -> None:
    st.warning(f"⚠ {msg}  \n`python src/pipeline.py` → `src/nlp/features.py` → `src/graph/pipeline.py`")


# ── Utility ───────────────────────────────────────────────────────────────────

def char_color(name: str) -> str:
    return CHAR_COLORS.get(name, DEFAULT_COLOR)


def episode_label(row) -> str:
    return f"S{int(row.season):02d}E{int(row.episode_num):02d}"


def tab_power():
    inf = load("suits_influence.csv")
    dom = load("suits_dominance.csv")

    if inf is None:
        _warn("suits_influence.csv not found.")
        return

    seasons = sorted(inf["season"].unique())
    chars   = [c for c in MAIN_CHARS if c in inf["character"].unique()]

    c1, c2 = st.columns([1, 3])
    with c1:
        sel_seasons = st.multiselect("Seasons", seasons, default=seasons, key="pw_s")
        sel_chars   = st.multiselect("Characters", chars, default=chars[:6], key="pw_c")

    df = inf[inf["season"].isin(sel_seasons) & inf["character"].isin(sel_chars)].copy()
    df["ep_label"] = df.apply(episode_label, axis=1)
    df = df.sort_values(["season", "episode_num"])

    # ── Influence trajectory ─────────────────────────────────────────────────
    st.subheader("Influence Score Over Time")
    fig = go.Figure()
    for char in sel_chars:
        cdf = df[df["character"] == char]
        if cdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=cdf["ep_label"], y=cdf["influence_score"],
            mode="lines+markers",
            name=char.split()[0],
            line=dict(color=char_color(char), width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{char}</b><br>Episode: %{{x}}<br>Influence: %{{y:.3f}}<extra></extra>",
        ))

    # Season separators — add_vline doesn't work on categorical axes, use shapes instead
    all_labels = df.sort_values(["season", "episode_num"])["ep_label"].unique().tolist()
    for season in sel_seasons[1:]:
        season_eps = df[df["season"] == season]["ep_label"]
        if season_eps.empty:
            continue
        first_ep = season_eps.iloc[0]
        if first_ep in all_labels:
            x_idx = all_labels.index(first_ep)
            fig.add_shape(type="line", xref="x", yref="paper",
                          x0=first_ep, x1=first_ep, y0=0, y1=1,
                          line=dict(dash="dot", color="rgba(255,255,255,0.15)", width=1))
            fig.add_annotation(x=first_ep, y=1.02, xref="x", yref="paper",
                               text=f"S{season}", showarrow=False,
                               font=dict(color="#aaa", size=10), xanchor="left")

    fig.update_layout(
        height=380, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(title="Influence Score", gridcolor="#1f2937"),
        margin=dict(t=20, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Season average rankings ───────────────────────────────────────────────
    st.subheader("Average Influence by Season")
    avg = (
        df.groupby(["season", "character"])["influence_score"]
        .mean().reset_index()
        .rename(columns={"influence_score": "avg_influence"})
    )
    fig2 = px.bar(
        avg, x="season", y="avg_influence", color="character",
        barmode="group",
        color_discrete_map={c: char_color(c) for c in sel_chars},
        labels={"avg_influence": "Avg Influence", "season": "Season"},
        template="plotly_dark",
    )
    fig2.update_layout(height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                       legend_title="", margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

    # Dominance count 
    if dom is not None:
        st.subheader("Episodes as Dominant Character")
        dom_filt = dom[dom["season"].isin(sel_seasons)]
        dom_count = (
            dom_filt[dom_filt["dominant_character"].isin(sel_chars)]
            .groupby("dominant_character")["episode_id"].count()
            .reset_index(name="dominant_episodes")
            .sort_values("dominant_episodes", ascending=False)
        )
        fig3 = px.bar(
            dom_count, x="dominant_character", y="dominant_episodes",
            color="dominant_character",
            color_discrete_map={c: char_color(c) for c in sel_chars},
            template="plotly_dark",
            labels={"dominant_character": "", "dominant_episodes": "Episodes"},
        )
        fig3.update_layout(height=280, showlegend=False, paper_bgcolor="#0e1117",
                           plot_bgcolor="#0e1117", margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)


def tab_network():
    ei  = load("suits_episode_interactions.csv")
    inf = load("suits_influence.csv")
    cen = load("suits_centrality.csv")

    if ei is None or inf is None:
        _warn("suits_episode_interactions.csv or suits_influence.csv not found.")
        return

    episodes = (
        ei[["episode_id", "season", "episode_num"]]
        .drop_duplicates()
        .sort_values(["season", "episode_num"])
    )
    ep_options = episodes["episode_id"].tolist()

    col1, col2 = st.columns([2, 1])
    with col1:
        ep_id = st.selectbox("Episode", ep_options, key="net_ep")
    with col2:
        min_weight = st.slider("Min edge weight", 0.0, 10.0, 0.5, 0.5, key="net_mw")

    # Build graph
    ep_ei = ei[(ei["episode_id"] == ep_id) & (ei["total_weight"] >= min_weight)]
    G = nx.Graph()
    for _, r in ep_ei.iterrows():
        G.add_edge(r["char_a"], r["char_b"], weight=float(r["total_weight"]),
                   scenes=int(r["scene_count"]))

    if G.number_of_nodes() == 0:
        st.info("No interactions found for this episode. Lower the min edge weight.")
        return

    # Layout
    pos = nx.spring_layout(G, weight="weight", seed=42,
                           k=1.5 / max(G.number_of_nodes() ** 0.5, 1))

    # Edge traces
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w_norm = data["weight"] / max_w
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=max(0.5, w_norm * 6), color=f"rgba(180,180,180,{0.2 + w_norm*0.5})"),
            hoverinfo="none", showlegend=False,
        ))

    # Node trace
    inf_ep = inf[inf["episode_id"] == ep_id].set_index("character")
    nx_list, ny_list, nt, ns, nc, nh = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        nx_list.append(x); ny_list.append(y)
        nt.append(node.split()[0])
        nc.append(char_color(node))
        row = inf_ep.loc[node] if node in inf_ep.index else None
        score = float(row["influence_score"]) if row is not None else 0.05
        pr    = float(row["pagerank"])        if row is not None else 0.0
        rank  = int(row["influence_rank"])    if row is not None else 99
        ns.append(max(18, score * 110))
        nh.append(f"<b>{node}</b><br>Influence: {score:.3f}<br>PageRank: {pr:.4f}<br>Rank: #{rank}")

    node_trace = go.Scatter(
        x=nx_list, y=ny_list, mode="markers+text",
        text=nt, textposition="top center",
        hovertext=nh, hoverinfo="text",
        marker=dict(size=ns, color=nc, line=dict(width=2, color="white")),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=520, template="plotly_dark",
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=10, b=10, l=10, r=10), hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Centrality table
    if cen is not None:
        st.subheader("Centrality Metrics")
        cen_ep = (
            cen[cen["episode_id"] == ep_id]
            .sort_values("pagerank", ascending=False)
            [["character", "pagerank", "degree_centrality",
              "betweenness_centrality", "eigenvector_centrality", "weighted_degree"]]
            .reset_index(drop=True)
        )
        cen_ep.columns = ["Character", "PageRank", "Degree", "Betweenness", "Eigenvector", "Wtd Degree"]
        st.dataframe(cen_ep.style.format({c: "{:.4f}" for c in cen_ep.columns[1:]}),
                     use_container_width=True, hide_index=True)


def tab_emotion():
    feat = load("suits_features.csv")
    inf  = load("suits_influence.csv")

    if feat is None:
        _warn("suits_features.csv not found.")
        return

    # Episode-level aggregates
    ep_agg = feat.groupby(["episode_id"]).agg(
        season=("season", "first"),
        episode_num=("episode_num", "first"),
        avg_sentiment=("vader_compound", "mean"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
        pct_commanding=("power_category", lambda x: (x == "commanding").mean()),
        pct_defensive=("power_category", lambda x: (x == "defensive").mean()),
        pct_manipulative=("power_category", lambda x: (x == "manipulative").mean()),
        pct_assertive=("power_category", lambda x: (x == "assertive").mean()),
        pct_anger=("emotion_label", lambda x: (x == "anger").mean()),
        pct_joy=("emotion_label", lambda x: (x == "joy").mean()),
        pct_fear=("emotion_label", lambda x: (x == "fear").mean()),
        pct_sadness=("emotion_label", lambda x: (x == "sadness").mean()),
        pct_surprise=("emotion_label", lambda x: (x == "surprise").mean()),
        pct_trust=("emotion_label", lambda x: (x == "trust").mean()),
        legal_density=("has_legal_language", "mean"),
    ).reset_index().sort_values(["season", "episode_num"])
    ep_agg["ep_label"] = ep_agg.apply(episode_label, axis=1)

    st.subheader("Sentiment Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ep_agg["ep_label"], y=ep_agg["avg_sentiment"].rolling(3, min_periods=1, center=True).mean(),
        mode="lines", name="Avg Sentiment", line=dict(color="#60A5FA", width=2.5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      xaxis=dict(tickangle=-45, showgrid=False),
                      yaxis=dict(title="VADER Compound", gridcolor="#1f2937"),
                      margin=dict(t=10, b=60))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Emotion Distribution")
        emo_cols = {"anger":"#EF4444", "joy":"#F59E0B", "fear":"#8B5CF6",
                    "sadness":"#6B7280", "surprise":"#06B6D4", "trust":"#10B981"}
        fig2 = go.Figure()
        for emo, color in emo_cols.items():
            col = f"pct_{emo}"
            if col in ep_agg:
                fig2.add_trace(go.Scatter(
                    x=ep_agg["ep_label"],
                    y=ep_agg[col].rolling(3, min_periods=1, center=True).mean(),
                    stackgroup="one", name=emo.title(),
                    line=dict(color=color, width=0),
                    fillcolor=color.rstrip(")") + ",0.7)" if color.startswith("rgb") else color,
                ))
        fig2.update_layout(height=300, template="plotly_dark", paper_bgcolor="#0e1117",
                           plot_bgcolor="#0e1117", xaxis=dict(showticklabels=False),
                           yaxis=dict(title="Fraction", gridcolor="#1f2937"),
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Power Language Breakdown")
        pow_cols = {"commanding":"#EF4444", "defensive":"#3B82F6",
                    "manipulative":"#8B5CF6", "assertive":"#10B981"}
        fig3 = go.Figure()
        for cat, color in pow_cols.items():
            col = f"pct_{cat}"
            if col in ep_agg:
                fig3.add_trace(go.Bar(
                    x=ep_agg["ep_label"], y=ep_agg[col],
                    name=cat.title(), marker_color=color,
                ))
        fig3.update_layout(barmode="stack", height=300, template="plotly_dark",
                           paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                           xaxis=dict(showticklabels=False),
                           yaxis=dict(title="Fraction", gridcolor="#1f2937"),
                           legend=dict(orientation="h", y=-0.15),
                           margin=dict(t=10, b=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Legal Language Density")
    fig4 = go.Figure(go.Scatter(
        x=ep_agg["ep_label"], y=ep_agg["legal_density"].rolling(3, min_periods=1).mean(),
        mode="lines", fill="tozeroy", line=dict(color="#F59E0B", width=2),
        fillcolor="rgba(245,158,11,0.15)",
    ))
    fig4.update_layout(height=200, template="plotly_dark", paper_bgcolor="#0e1117",
                       plot_bgcolor="#0e1117",
                       xaxis=dict(tickangle=-45, showgrid=False),
                       yaxis=dict(title="Fraction of lines", gridcolor="#1f2937"),
                       margin=dict(t=10, b=60))
    st.plotly_chart(fig4, use_container_width=True)



def tab_story():
    evo  = load("suits_topic_evolution.csv")
    kw   = load("suits_topic_keywords.csv")

    if evo is None or evo.empty:
        st.info("Topic model not yet run.  \n`python src/topics/pipeline.py`")
        return

    evo["ep_label"] = evo.apply(episode_label, axis=1)
    topic_cols = sorted([c for c in evo.columns if c.startswith("topic_") and "_smooth" not in c
                         and c.replace("topic_", "").isdigit()])
    smooth_cols = [c + "_smooth" for c in topic_cols if c + "_smooth" in evo.columns]
    use_cols = smooth_cols if smooth_cols else topic_cols

    # Get dominant topic labels per topic
    labels = {}
    if kw is not None and not kw.empty:
        for tid in kw["topic_id"].unique():
            top3 = kw[kw["topic_id"] == tid]["keyword"].head(3).tolist()
            labels[tid] = f"T{tid}: {', '.join(top3)}"
    else:
        for i, col in enumerate(topic_cols):
            labels[i] = f"Topic {i}"

    PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel

    st.subheader("Topic Intensity Over Time")
    fig = go.Figure()
    for i, (col, scol) in enumerate(zip(topic_cols, use_cols)):
        tid = int(col.replace("topic_", ""))
        fig.add_trace(go.Scatter(
            x=evo["ep_label"], y=evo[scol],
            mode="lines", name=labels.get(tid, f"T{tid}"),
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="#0e1117",
                      plot_bgcolor="#0e1117",
                      xaxis=dict(tickangle=-45, showgrid=False),
                      yaxis=dict(title="Topic Weight", gridcolor="#1f2937"),
                      legend=dict(orientation="h", y=-0.25, font_size=11),
                      margin=dict(t=10, b=100))
    st.plotly_chart(fig, use_container_width=True)

    # Dominant topic timeline (colour strip)
    st.subheader("Dominant Narrative Arc per Episode")
    dom_map = {int(c.replace("topic_", "")): labels.get(int(c.replace("topic_", "")), c)
               for c in topic_cols}
    evo["dom_label"] = evo["dominant_topic"].map(dom_map)
    fig2 = px.bar(evo, x="ep_label", y=[1]*len(evo), color="dom_label",
                  color_discrete_sequence=PALETTE, template="plotly_dark",
                  labels={"ep_label": "Episode", "y": "", "dom_label": "Topic"})
    fig2.update_layout(height=200, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                       yaxis=dict(showticklabels=False, showgrid=False),
                       xaxis=dict(tickangle=-45, showgrid=False),
                       showlegend=True, legend=dict(orientation="h", y=-0.35),
                       margin=dict(t=5, b=80))
    st.plotly_chart(fig2, use_container_width=True)

    # Keywords table
    if kw is not None and not kw.empty:
        with st.expander("Topic Keywords", expanded=False):
            pivot = kw.pivot_table(index="rank", columns="topic_id", values="keyword", aggfunc="first")
            pivot.columns = [labels.get(c, f"T{c}") for c in pivot.columns]
            st.dataframe(pivot, use_container_width=True)



def tab_predictor():
    feat_df = load("suits_predict_features.csv")
    dom     = load("suits_dominance.csv")
    eval_df = load("suits_eval_results.csv")

    if feat_df is None:
        _warn("suits_predict_features.csv not found. Run `python src/predict/pipeline.py`")
        return

    episodes = (
        feat_df[["episode_id", "season", "episode_num"]]
        .drop_duplicates()
        .sort_values(["season", "episode_num"])
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        ep_id = st.selectbox("Predict dominant character AFTER:", episodes["episode_id"].tolist(), key="pred_ep")
    with col2:
        model_choice = st.selectbox("Model", ["random_forest", "lstm"], key="pred_model")

    # Load model
    model_dir = DATA_DIR.parent / "models"
    model_path = model_dir / ("rf_model.pkl" if model_choice == "random_forest" else "lstm_model.pt")

    if not model_path.exists():
        st.info(f"No trained model found at `{model_path}`.  \n"
                "Run `python src/predict/pipeline.py` on the full dataset first.")
        _demo_prediction(ep_id, dom)
        return



    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from predict.models import RandomForestPredictor, LSTMPredictor
        from predict.features import build_lstm_matrix

        if model_choice == "random_forest":
            model = RandomForestPredictor.load(model_path)
            ranked = model.predict_proba_dominance(feat_df, ep_id)
        else:
            model = LSTMPredictor.load(model_path)
            X_wide, _ = build_lstm_matrix(feat_df)
            ranked = model.predict_proba_dominance(X_wide, ep_id)

    except Exception as e:
        st.error(f"Model load failed: {e}")
        _demo_prediction(ep_id, dom)
        return

    if not ranked:
        st.warning("No prediction available for this episode.")
        return

    _show_prediction(ranked, ep_id, dom)

    if eval_df is not None and not eval_df.empty:
        st.divider()
        st.subheader("Model Evaluation (held-out episodes)")
        st.dataframe(eval_df.style.format({
            "top1_accuracy": "{:.1%}", "top3_accuracy": "{:.1%}"
        }), use_container_width=True, hide_index=True)


def _show_prediction(ranked: dict[str, float], ep_id: str, dom) -> None:
    chars = list(ranked.keys())[:6]
    probs = [ranked[c] for c in chars]
    pred  = chars[0]

    st.subheader(f"Predicted dominant character after {ep_id}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <p style="color:#aaa;font-size:13px;margin:0">PREDICTED</p>
          <p style="color:{char_color(pred)};font-size:28px;font-weight:700;margin:4px 0">{pred.split()[0]}</p>
          <p style="color:#60A5FA;font-size:20px;margin:0">{probs[0]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=probs, y=chars, orientation="h",
            marker_color=[char_color(c) for c in chars],
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117",
                          xaxis=dict(tickformat=".0%", range=[0, max(probs)*1.1]),
                          yaxis=dict(autorange="reversed"),
                          margin=dict(t=5, b=5, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    if dom is not None:
        ep_ids = sorted(dom["episode_id"].unique())
        if ep_id in ep_ids:
            idx = ep_ids.index(ep_id)
            if idx + 1 < len(ep_ids):
                next_ep = ep_ids[idx + 1]
                actual = dom[dom["episode_id"] == next_ep]["dominant_character"]
                if not actual.empty:
                    actual_char = actual.iloc[0]
                    correct = actual_char == pred
                    icon = "✅" if correct else "❌"
                    st.info(f"{icon} Actual dominant in {next_ep}: **{actual_char}**")


def _demo_prediction(ep_id: str, dom) -> None:
    """Show a plausible-looking prediction using influence scores as proxy."""
    inf = load("suits_influence.csv")
    if inf is None:
        return
    ep_inf = inf[inf["episode_id"] == ep_id].copy()
    if ep_inf.empty:
        return
    ep_inf = ep_inf.sort_values("influence_score", ascending=False).head(6)
    total = ep_inf["influence_score"].sum()
    ranked = {row["character"]: round(row["influence_score"]/total, 3)
              for _, row in ep_inf.iterrows()}
    _show_prediction(ranked, ep_id, dom)


def main():
    st.title("⚖️ Suits Power Network")
    st.caption("Modeling narrative control, influence, and power shifts across 9 seasons")
    st.divider()

    t1, t2, t3, t4, t5 = st.tabs([
        "⚡ Character Power",
        "🕸️ Network Graph",
        "😤 Emotion Tracker",
        "📖 Story Evolution",
        "🔮 Dominance Predictor",
    ])

    with t1: tab_power()
    with t2: tab_network()
    with t3: tab_emotion()
    with t4: tab_story()
    with t5: tab_predictor()


main()
