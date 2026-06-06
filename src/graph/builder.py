"""
Graph Builder — Step 3.1 / 3.2

From suits_dialogue.csv (which has char_mentions and scene_id per line),
this module computes:
  1. scene-level character co-occurrence
  2. episode-level interaction weights   → suits_episode_interactions.csv
  3. one NetworkX weighted graph per episode  G₁, G₂, … G₁₃₄
  4. a cumulative "all-seasons" graph         G_all

Edge weight formula (from SPEC):
  interaction_strength = co_occurrence_frequency + dialogue_exchange_count
  Here we use geometric-mean of mention counts within shared scenes, summed
  across all scenes per episode. This encodes both frequency and intensity.

Nodes: canonical character full names  (e.g. "Harvey Specter")
Edges: weighted undirected
Edge attributes: weight (float), scene_count (int)
"""

from itertools import combinations

import networkx as nx
import pandas as pd


# ── Co-occurrence computation ─────────────────────────────────────────────────

def compute_episode_interactions(dialogue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw dialogue rows (with pipe-separated char_mentions per line)
    into episode-level interaction weights between character pairs.

    Returns DataFrame with columns:
      episode_id, season, episode_num, char_a, char_b, total_weight, scene_count
    """
    scene_rows = []

    for scene_id, scene_grp in dialogue_df.groupby("scene_id", sort=False):
        episode_id = scene_grp["episode_id"].iloc[0]
        season     = scene_grp["season"].iloc[0]
        ep_num     = scene_grp["episode_num"].iloc[0]

        # Aggregate mention counts per character across all lines in the scene
        char_counts: dict[str, int] = {}
        for mentions_str in scene_grp["char_mentions"].dropna():
            if not mentions_str:
                continue
            for char in str(mentions_str).split("|"):
                char = char.strip()
                if char:
                    char_counts[char] = char_counts.get(char, 0) + 1

        if len(char_counts) < 2:
            continue

        for a, b in combinations(sorted(char_counts), 2):
            scene_rows.append({
                "scene_id":   scene_id,
                "episode_id": episode_id,
                "season":     season,
                "episode_num": ep_num,
                "char_a":     a,
                "char_b":     b,
                "weight":     (char_counts[a] * char_counts[b]) ** 0.5,
            })

    if not scene_rows:
        return pd.DataFrame()

    scene_df = pd.DataFrame(scene_rows)

    # Aggregate to episode level
    ep_df = (
        scene_df
        .groupby(["episode_id", "season", "episode_num", "char_a", "char_b"])
        .agg(total_weight=("weight", "sum"), scene_count=("scene_id", "count"))
        .reset_index()
        .sort_values(["episode_id", "total_weight"], ascending=[True, False])
    )
    return ep_df


# ── NetworkX graph construction ───────────────────────────────────────────────

def _make_graph(ep_id: str, season: int, ep_num: int, rows: pd.DataFrame) -> nx.Graph:
    G = nx.Graph(episode_id=ep_id, season=season, episode_num=ep_num)
    for _, r in rows.iterrows():
        G.add_edge(
            r["char_a"], r["char_b"],
            weight=float(r["total_weight"]),
            scene_count=int(r["scene_count"]),
        )
    return G


def build_temporal_graphs(ep_interactions: pd.DataFrame) -> dict[str, nx.Graph]:
    """
    Build one weighted NetworkX graph per episode.
    Returns ordered dict  { episode_id → nx.Graph }  sorted S01E01 … S09E10.
    """
    graphs: dict[str, nx.Graph] = {}

    for ep_id, grp in ep_interactions.groupby("episode_id", sort=True):
        season = int(grp["season"].iloc[0])
        ep_num = int(grp["episode_num"].iloc[0])
        graphs[ep_id] = _make_graph(ep_id, season, ep_num, grp)

    return graphs


def build_cumulative_graph(
    ep_interactions: pd.DataFrame,
    up_to_season: int | None = None,
) -> nx.Graph:
    """
    Cumulative graph across all episodes (or up to a given season).
    Edge weights are summed across all contributing episodes.
    """
    df = ep_interactions
    if up_to_season is not None:
        df = df[df["season"] <= up_to_season]

    G = nx.Graph(cumulative=True, up_to_season=up_to_season)
    for _, r in df.iterrows():
        a, b, w, s = r["char_a"], r["char_b"], r["total_weight"], r["scene_count"]
        if G.has_edge(a, b):
            G[a][b]["weight"]      += w
            G[a][b]["scene_count"] += s
        else:
            G.add_edge(a, b, weight=float(w), scene_count=int(s))
    return G


# ── Graph stats helper (used by pipeline for printing) ───────────────────────

def graph_summary(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    rows = []
    for ep_id, G in graphs.items():
        rows.append({
            "episode_id":  ep_id,
            "season":      G.graph.get("season"),
            "episode_num": G.graph.get("episode_num"),
            "n_nodes":     G.number_of_nodes(),
            "n_edges":     G.number_of_edges(),
            "total_weight": sum(d["weight"] for _, _, d in G.edges(data=True)),
            "density":     nx.density(G),
        })
    return pd.DataFrame(rows).sort_values(["season", "episode_num"])
