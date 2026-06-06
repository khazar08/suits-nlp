"""
Prediction Feature Engineering — Step 5.1

Builds per-(episode, character) feature vectors for training the dominance predictor.

Target (binary):  is_next_dominant — 1 if this character dominates episode T+1
Target (multi):   next_dominant    — name of the character who dominates T+1

Feature groups:
  Graph features        pagerank, degree, betweenness, eigenvector, weighted_degree
  Influence features    influence_score, mention_volume_share, sentiment_control_norm
  Lag features          above metrics at T-1, T-2, T-3
  Momentum              influence_score[T] - influence_score[T-1]
  Dominance streak      consecutive episodes as dominant character
  Episode NLP           avg sentiment, power language fractions, legal density, emotions
  Episode metadata      season, position, is_finale, is_premiere, season_progress
"""

import numpy as np
import pandas as pd

# Characters we track in the wide LSTM feature matrix
MAIN_CHARS = [
    "Harvey Specter", "Mike Ross", "Louis Litt", "Donna Paulsen",
    "Jessica Pearson", "Rachel Zane", "Katrina Bennett", "Alex Williams",
    "Samantha Wheeler",
]

LAG_PERIODS = [1, 2, 3]
CHAR_FEATURES = [
    "influence_score", "pagerank", "degree_centrality",
    "betweenness_centrality", "eigenvector_centrality",
    "weighted_degree", "mention_volume_share",
]


# ── Episode-level NLP aggregates ──────────────────────────────────────────────

def _build_episode_nlp(features_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line-level NLP features to one row per episode."""
    df = features_df.copy()
    agg = df.groupby("episode_id").agg(
        avg_sentiment       =("vader_compound",    "mean"),
        pct_positive        =("sentiment_label",   lambda x: (x == "positive").mean()),
        pct_negative        =("sentiment_label",   lambda x: (x == "negative").mean()),
        pct_commanding      =("power_category",    lambda x: (x == "commanding").mean()),
        pct_defensive       =("power_category",    lambda x: (x == "defensive").mean()),
        pct_manipulative    =("power_category",    lambda x: (x == "manipulative").mean()),
        pct_assertive       =("power_category",    lambda x: (x == "assertive").mean()),
        legal_density       =("has_legal_language","mean"),
        avg_word_count      =("word_count",        "mean"),
        avg_flesch          =("flesch_ease",       "mean"),
        pct_anger           =("emotion_label",     lambda x: (x == "anger").mean()),
        pct_fear            =("emotion_label",     lambda x: (x == "fear").mean()),
        pct_joy             =("emotion_label",     lambda x: (x == "joy").mean()),
        pct_sadness         =("emotion_label",     lambda x: (x == "sadness").mean()),
    ).reset_index()
    return agg.round(6)


# ── Dominance streak ──────────────────────────────────────────────────────────

def _compute_streaks(df: pd.DataFrame, dominance_df: pd.DataFrame) -> pd.Series:
    """Consecutive episodes each character was dominant, as of episode T."""
    dom_map = dominance_df.set_index("episode_id")["dominant_character"].to_dict()

    streaks = []
    for (char,), grp in df.groupby(["character"], sort=False):
        grp = grp.sort_values(["season", "episode_num"])
        count = 0
        for ep_id in grp["episode_id"]:
            count = count + 1 if dom_map.get(ep_id) == char else 0
            streaks.append((grp.index[grp["episode_id"] == ep_id][0], count))

    streak_series = pd.Series({idx: val for idx, val in streaks}, name="dominance_streak")
    return streak_series


# ── Main feature builder ──────────────────────────────────────────────────────

def build_feature_matrix(
    influence_df:   pd.DataFrame,
    centrality_df:  pd.DataFrame,
    nlp_features_df: pd.DataFrame,
    dominance_df:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (episode, character), including
    lag features, momentum, streak, NLP aggregates, and the binary target.

    Requires at least 4 episodes of data (for lag-3 features).
    """
    # ── Merge influence + centrality ─────────────────────────────────────────
    cent_cols = ["episode_id", "character", "degree_centrality",
                 "betweenness_centrality", "eigenvector_centrality", "weighted_degree"]
    df = influence_df.merge(
        centrality_df[cent_cols], on=["episode_id", "character"], how="left"
    )

    # ── Episode metadata ─────────────────────────────────────────────────────
    ep_order = (
        df[["episode_id", "season", "episode_num"]]
        .drop_duplicates()
        .sort_values(["season", "episode_num"])
        .reset_index(drop=True)
    )
    ep_order["episode_idx"]       = ep_order.index
    ep_order["season_max_ep"]     = ep_order.groupby("season")["episode_num"].transform("max")
    ep_order["is_season_finale"]  = (ep_order["episode_num"] == ep_order["season_max_ep"]).astype(int)
    ep_order["is_season_premiere"]= (ep_order["episode_num"] == 1).astype(int)
    ep_order["season_progress"]   = (ep_order["episode_num"] / ep_order["season_max_ep"]).round(4)
    df = df.merge(
        ep_order.drop(columns="season_max_ep"),
        on=["episode_id", "season", "episode_num"], how="left"
    )

    # ── Sort for lag computation ─────────────────────────────────────────────
    df = df.sort_values(["character", "season", "episode_num"]).reset_index(drop=True)

    # ── Lag features ────────────────────────────────────────────────────────
    for col in CHAR_FEATURES:
        if col not in df.columns:
            continue
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = df.groupby("character")[col].shift(lag)

    # ── Momentum & rank change ───────────────────────────────────────────────
    df["influence_momentum"]   = df["influence_score"] - df["influence_score_lag1"].fillna(0)
    df["pagerank_momentum"]    = df["pagerank"] - df["pagerank_lag1"].fillna(0)
    df["rank_momentum"]        = df.groupby("character")["influence_rank"].shift(1) - df["influence_rank"]

    # ── Dominance streak ─────────────────────────────────────────────────────
    streaks = _compute_streaks(df, dominance_df)
    df["dominance_streak"] = streaks

    # ── Episode NLP aggregates ────────────────────────────────────────────────
    ep_nlp = _build_episode_nlp(nlp_features_df)
    df = df.merge(ep_nlp, on="episode_id", how="left")

    # ── Target: is this character dominant in episode T+1? ───────────────────
    ep_ids = ep_order["episode_id"].tolist()
    next_ep = {ep_ids[i]: ep_ids[i + 1] for i in range(len(ep_ids) - 1)}
    dom_map = dominance_df.set_index("episode_id")["dominant_character"].to_dict()

    df["next_episode_id"]    = df["episode_id"].map(next_ep)
    df["next_dominant"]      = df["next_episode_id"].map(dom_map)
    df["is_next_dominant"]   = (df["character"] == df["next_dominant"]).astype(int)

    return df


# ── Wide matrix for LSTM (one row per episode) ────────────────────────────────

def build_lstm_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a wide (episode × characters×features) matrix for the LSTM.
    One row per episode; columns = feature × character for each MAIN_CHAR.

    Returns (X, y) where y is the index into MAIN_CHARS of the next dominant.
    """
    ep_order = (
        feature_df[["episode_id", "season", "episode_num"]]
        .drop_duplicates()
        .sort_values(["season", "episode_num"])
        ["episode_id"].tolist()
    )

    cols_per_char = CHAR_FEATURES + [f"{c}_lag1" for c in CHAR_FEATURES] + [
        "influence_momentum", "dominance_streak",
    ]

    rows = []
    targets = []

    for ep_id in ep_order:
        ep_df = feature_df[feature_df["episode_id"] == ep_id]
        row = {}
        for char in MAIN_CHARS:
            char_row = ep_df[ep_df["character"] == char]
            for col in cols_per_char:
                key = f"{char.split()[0]}_{col}"
                row[key] = float(char_row[col].iloc[0]) if not char_row.empty and col in char_row.columns and not char_row[col].isna().all() else 0.0
        rows.append(row)

        # Target
        next_dom = ep_df["next_dominant"].dropna()
        if not next_dom.empty and next_dom.iloc[0] in MAIN_CHARS:
            targets.append(MAIN_CHARS.index(next_dom.iloc[0]))
        else:
            targets.append(-1)

    X = pd.DataFrame(rows, index=ep_order).fillna(0)
    y = pd.Series(targets, index=ep_order, name="next_dominant_idx")

    return X, y


# ── Feature selection helpers ─────────────────────────────────────────────────

FEATURE_COLS = [
    # Graph / influence
    "pagerank", "degree_centrality", "betweenness_centrality",
    "eigenvector_centrality", "weighted_degree",
    "influence_score", "mention_volume_share", "sentiment_control_norm",
    "influence_rank",
    # Lag
    "influence_score_lag1", "influence_score_lag2", "influence_score_lag3",
    "pagerank_lag1", "pagerank_lag2",
    "mention_volume_share_lag1",
    # Momentum
    "influence_momentum", "pagerank_momentum", "rank_momentum",
    "dominance_streak",
    # Episode NLP
    "avg_sentiment", "pct_positive", "pct_negative",
    "pct_commanding", "pct_defensive", "pct_manipulative", "pct_assertive",
    "legal_density", "avg_word_count", "avg_flesch",
    "pct_anger", "pct_fear", "pct_joy", "pct_sadness",
    # Metadata
    "season", "episode_idx", "season_progress",
    "is_season_finale", "is_season_premiere",
]


def get_X_y(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract model-ready feature matrix X and binary target y."""
    available = [c for c in FEATURE_COLS if c in feature_df.columns]
    # Drop rows with no lag data (first 3 episodes per character)
    mask = feature_df["next_dominant"].notna() & feature_df["influence_score_lag1"].notna()
    df = feature_df[mask].copy()
    X = df[available].fillna(0)
    y = df["is_next_dominant"]
    return X, y
