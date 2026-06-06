"""
Centrality + Dynamic Influence Score — Step 3 / 4

Centrality metrics computed per episode graph:
  degree_centrality     — fraction of possible connections used
  betweenness_centrality— fraction of all shortest paths through this node
  eigenvector_centrality— weighted score from being connected to high scorers
  pagerank              — authority score (who controls the network)

Dynamic Influence Score (SPEC formula):
  Influence = 0.4 × PageRank_norm
            + 0.3 × Mention_Volume_Share
            + 0.3 × Sentiment_Control_norm

  PageRank_norm          — min-max normalized within episode
  Mention_Volume_Share   — character's mention count / episode total mentions
  Sentiment_Control_norm — normalized mean |vader_compound| of scenes
                           where character appears (from suits_features.csv)

Outputs:
  suits_centrality.csv       — all four centrality metrics per (episode, char)
  suits_influence.csv        — influence scores + per-episode ranks
  suits_power_trajectory.csv — influence over time, with rank_change column
  suits_dominance.csv        — dominant character per episode
"""

import warnings
import networkx as nx
import pandas as pd
import numpy as np


# ── Centrality ────────────────────────────────────────────────────────────────

def compute_centrality(
    graphs: dict[str, nx.Graph],
) -> pd.DataFrame:
    """
    Compute degree, betweenness, eigenvector, and PageRank for every
    episode graph. Returns a long-format DataFrame.
    """
    rows = []

    for ep_id, G in graphs.items():
        n = G.number_of_nodes()
        if n < 2:
            continue

        season  = G.graph.get("season", 0)
        ep_num  = G.graph.get("episode_num", 0)

        degree      = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eigenvector = nx.eigenvector_centrality(
                        G, weight="weight", max_iter=500, tol=1e-4
                    )
            except Exception:
                eigenvector = {v: 0.0 for v in G.nodes()}

        pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

        for char in G.nodes():
            rows.append({
                "episode_id":              ep_id,
                "season":                  season,
                "episode_num":             ep_num,
                "character":               char,
                "degree_centrality":       round(degree.get(char, 0), 6),
                "betweenness_centrality":  round(betweenness.get(char, 0), 6),
                "eigenvector_centrality":  round(eigenvector.get(char, 0), 6),
                "pagerank":                round(pagerank.get(char, 0), 6),
                # Weighted degree (raw interaction strength)
                "weighted_degree":         round(
                    sum(d["weight"] for _, _, d in G.edges(char, data=True)), 3
                ),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["season", "episode_num", "pagerank"], ascending=[True, True, False])
    return df


# ── Influence Score ───────────────────────────────────────────────────────────

def _minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo + 1e-9)


def compute_influence(
    centrality_df: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    features_df: pd.DataFrame,
    w_pagerank: float = 0.4,
    w_volume:   float = 0.3,
    w_sentiment: float = 0.3,
) -> pd.DataFrame:
    """
    Compute the dynamic influence score per (episode, character).

    dialogue_df  — suits_dialogue.csv  (has char_mentions, scene_id, episode_id)
    features_df  — suits_features.csv  (has scene_id, vader_compound)
    """

    # ── Component 1: mention volume share ────────────────────────────────────
    char_mentions = (
        dialogue_df[dialogue_df["char_mentions"].notna() & (dialogue_df["char_mentions"] != "")]
        .assign(chars=lambda d: d["char_mentions"].str.split("|"))
        .explode("chars")
        .assign(chars=lambda d: d["chars"].str.strip())
        .query("chars != ''")
        .groupby(["episode_id", "chars"])["mention_count"]
        .sum()
        .reset_index()
        .rename(columns={"chars": "character", "mention_count": "char_ep_mentions"})
    )

    ep_totals = char_mentions.groupby("episode_id")["char_ep_mentions"].sum().rename("ep_total_mentions")
    char_mentions = char_mentions.join(ep_totals, on="episode_id")
    char_mentions["mention_volume_share"] = (
        char_mentions["char_ep_mentions"] / char_mentions["ep_total_mentions"]
    ).round(6)

    # ── Component 2: sentiment control ───────────────────────────────────────
    # For each scene, compute mean |vader_compound| across all lines
    scene_sentiment = (
        features_df.groupby(["episode_id", "scene_id"])["vader_compound"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .rename(columns={"vader_compound": "scene_abs_sentiment"})
    )

    # For each (episode, character), get scenes they appear in and average sentiment
    scene_chars = (
        dialogue_df[dialogue_df["char_mentions"].notna() & (dialogue_df["char_mentions"] != "")]
        .assign(chars=lambda d: d["char_mentions"].str.split("|"))
        .explode("chars")
        .assign(chars=lambda d: d["chars"].str.strip())
        .query("chars != ''")
        [["episode_id", "scene_id", "chars"]]
        .drop_duplicates()
        .rename(columns={"chars": "character"})
        .merge(scene_sentiment, on=["episode_id", "scene_id"], how="left")
        .groupby(["episode_id", "character"])["scene_abs_sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"scene_abs_sentiment": "sentiment_control_raw"})
    )

    # ── Merge all components onto centrality_df ───────────────────────────────
    df = centrality_df.copy()
    df = df.merge(char_mentions[["episode_id", "character", "mention_volume_share"]], on=["episode_id", "character"], how="left")
    df = df.merge(scene_chars, on=["episode_id", "character"], how="left")
    df["mention_volume_share"]   = df["mention_volume_share"].fillna(0)
    df["sentiment_control_raw"]  = df["sentiment_control_raw"].fillna(0)

    # ── Normalize within each episode ────────────────────────────────────────
    df["pagerank_norm"]        = df.groupby("episode_id")["pagerank"].transform(_minmax_norm)
    df["sentiment_control_norm"] = df.groupby("episode_id")["sentiment_control_raw"].transform(_minmax_norm)

    # ── Composite influence score ────────────────────────────────────────────
    df["influence_score"] = (
        w_pagerank   * df["pagerank_norm"]
        + w_volume   * df["mention_volume_share"]
        + w_sentiment * df["sentiment_control_norm"]
    ).round(6)

    # ── Per-episode rank (1 = most influential) ───────────────────────────────
    df["influence_rank"] = df.groupby("episode_id")["influence_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    return df.sort_values(["season", "episode_num", "influence_rank"])


# ── Power trajectory & shift detection ───────────────────────────────────────

def compute_power_trajectory(influence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Track each character's influence rank episode by episode.
    Adds rank_change = previous_rank - current_rank
    (positive = rising, negative = falling).
    """
    df = influence_df[
        ["episode_id", "season", "episode_num", "character",
         "influence_score", "influence_rank"]
    ].copy().sort_values(["character", "season", "episode_num"])

    df["rank_change"] = df.groupby("character")["influence_rank"].diff(-1)  # positive = rising
    df["rank_change"] = df["rank_change"].fillna(0).astype(int)

    # Rolling 3-episode smoothed influence
    df["influence_smooth"] = (
        df.groupby("character")["influence_score"]
        .transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
        .round(6)
    )

    return df.sort_values(["season", "episode_num", "influence_rank"])


def compute_dominance(influence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per episode: the dominant character + top-3 list.
    Also flags episodes where the dominant character changed.
    """
    top1 = (
        influence_df[influence_df["influence_rank"] == 1]
        [["episode_id", "season", "episode_num", "character", "influence_score"]]
        .rename(columns={"character": "dominant_character", "influence_score": "dominant_score"})
    )

    top3 = (
        influence_df[influence_df["influence_rank"] <= 3]
        .sort_values(["episode_id", "influence_rank"])
        .groupby("episode_id")["character"]
        .apply(lambda chars: " | ".join(chars))
        .reset_index()
        .rename(columns={"character": "top3_characters"})
    )

    dom = top1.merge(top3, on="episode_id").sort_values(["season", "episode_num"])
    dom["power_shift"] = dom["dominant_character"] != dom["dominant_character"].shift(1)

    return dom


def detect_power_shifts(trajectory_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identify the biggest single-episode rank jumps for the top-N characters.
    Returns rows sorted by abs(rank_change) descending.
    """
    top_chars = (
        trajectory_df.groupby("character")["influence_score"].mean()
        .nlargest(top_n).index.tolist()
    )
    return (
        trajectory_df[trajectory_df["character"].isin(top_chars)]
        .assign(abs_change=lambda d: d["rank_change"].abs())
        .sort_values("abs_change", ascending=False)
        .head(20)
        [["episode_id", "season", "episode_num", "character",
          "influence_rank", "rank_change", "influence_score"]]
    )
