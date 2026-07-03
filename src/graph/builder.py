from itertools import combinations
import networkx as nx
import pandas as pd


def compute_episode_interactions(dialogue_df: pd.DataFrame) -> pd.DataFrame:
    scene_rows = []

    for scene_id, scene_grp in dialogue_df.groupby("scene_id", sort=False):
        episode_id = scene_grp["episode_id"].iloc[0]
        season     = scene_grp["season"].iloc[0]
        ep_num     = scene_grp["episode_num"].iloc[0]

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
    ep_df = (
        scene_df
        .groupby(["episode_id", "season", "episode_num", "char_a", "char_b"])
        .agg(total_weight=("weight", "sum"), scene_count=("scene_id", "count"))
        .reset_index()
        .sort_values(["episode_id", "total_weight"], ascending=[True, False])
    )
    return ep_df


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
