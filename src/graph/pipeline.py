"""
Graph Pipeline — Step 3 & 4 Orchestrator

Reads:
  data/processed/suits_dialogue.csv
  data/processed/suits_features.csv

Writes to data/processed/:
  suits_episode_interactions.csv  — edge weights per (episode, char_a, char_b)
  suits_centrality.csv            — 4 centrality metrics per (episode, character)
  suits_influence.csv             — dynamic influence score + per-episode ranks
  suits_power_trajectory.csv      — influence over time with rank_change
  suits_dominance.csv             — dominant character per episode + power shifts

Usage:
    python src/graph/pipeline.py
    python src/graph/pipeline.py --data-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.builder import (
    compute_episode_interactions,
    build_temporal_graphs,
    build_cumulative_graph,
    graph_summary,
)
from graph.metrics import (
    compute_centrality,
    compute_influence,
    compute_power_trajectory,
    compute_dominance,
    detect_power_shifts,
)


def run(data_dir: Path) -> dict:
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    _h("Loading data")
    dialogue_path = out_dir / "suits_dialogue.csv"
    features_path = out_dir / "suits_features.csv"

    if not dialogue_path.exists():
        sys.exit(f"[ERROR] {dialogue_path} not found — run pipeline.py first.")
    if not features_path.exists():
        sys.exit(f"[ERROR] {features_path} not found — run src/nlp/features.py first.")

    dialogue_df = pd.read_csv(dialogue_path)
    features_df = pd.read_csv(features_path)
    print(f"  dialogue:  {len(dialogue_df):,} lines  |  {dialogue_df['episode_id'].nunique()} episodes")
    print(f"  features:  {len(features_df):,} rows  |  {features_df.columns.tolist().count('vader_compound')} sentiment cols")

    # ── Step 3.1: Episode interaction weights ────────────────────────────────
    _h("STEP 3.1 — Computing episode interaction weights")
    ep_interactions = compute_episode_interactions(dialogue_df)
    if ep_interactions.empty:
        sys.exit("[ERROR] No character co-occurrences found. Check char_mentions column.")

    ep_interactions.to_csv(out_dir / "suits_episode_interactions.csv", index=False)
    print(f"  {len(ep_interactions):,} episode-level edges")
    print(f"  {ep_interactions['episode_id'].nunique()} episodes with interaction data")

    # ── Step 3.2: Temporal graphs G₁ … Gₙ ──────────────────────────────────
    _h("STEP 3.2 — Building temporal graphs G₁ … Gₙ")
    graphs = build_temporal_graphs(ep_interactions)
    summary = graph_summary(graphs)
    print(f"  Built {len(graphs)} episode graphs")
    print(f"  Avg nodes per graph: {summary['n_nodes'].mean():.1f}")
    print(f"  Avg edges per graph: {summary['n_edges'].mean():.1f}")
    print(f"  Avg density:         {summary['density'].mean():.3f}")

    # Cumulative graph
    G_all = build_cumulative_graph(ep_interactions)
    print(f"\n  Cumulative graph: {G_all.number_of_nodes()} nodes, {G_all.number_of_edges()} edges")

    # ── Step 4.1: Centrality metrics ─────────────────────────────────────────
    _h("STEP 4.1 — Computing centrality metrics")
    centrality_df = compute_centrality(graphs)
    centrality_df.to_csv(out_dir / "suits_centrality.csv", index=False)
    print(f"  Saved {len(centrality_df):,} rows → suits_centrality.csv")

    # Per-character PageRank averages (all seasons)
    avg_pr = (
        centrality_df.groupby("character")["pagerank"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\n  Top 10 characters by avg PageRank (all episodes):")
    for char, pr in avg_pr.items():
        bar = "█" * int(pr * 400)
        print(f"    {char:<25} {pr:.4f}  {bar}")

    # ── Step 4.2: Dynamic influence score ────────────────────────────────────
    _h("STEP 4.2 — Computing dynamic influence score")
    influence_df = compute_influence(centrality_df, dialogue_df, features_df)
    influence_df.to_csv(out_dir / "suits_influence.csv", index=False)
    print(f"  Saved {len(influence_df):,} rows → suits_influence.csv")

    # ── Power trajectory ─────────────────────────────────────────────────────
    _h("STEP 4.3 — Power trajectory over time")
    trajectory_df = compute_power_trajectory(influence_df)
    trajectory_df.to_csv(out_dir / "suits_power_trajectory.csv", index=False)
    print(f"  Saved → suits_power_trajectory.csv")

    # ── Episode dominance ────────────────────────────────────────────────────
    dominance_df = compute_dominance(influence_df)
    dominance_df.to_csv(out_dir / "suits_dominance.csv", index=False)

    # ── Summary report ───────────────────────────────────────────────────────
    _h("POWER GRAPH SUMMARY")

    # Dominant character per season
    season_dom = (
        dominance_df.groupby("season")["dominant_character"]
        .agg(lambda x: x.value_counts().index[0])  # mode
    )
    print("  Most dominant character per season:")
    for season, char in season_dom.items():
        ep_count = dominance_df[
            (dominance_df["season"] == season) &
            (dominance_df["dominant_character"] == char)
        ].shape[0]
        print(f"    Season {season}: {char:<25} (dominant in {ep_count} episodes)")

    # Power shifts
    shifts = detect_power_shifts(trajectory_df, top_n=6)
    n_shifts = dominance_df["power_shift"].sum()
    print(f"\n  Total power shifts detected: {n_shifts}")
    if not shifts.empty:
        print("  Biggest single-episode rank jumps:")
        for _, r in shifts.head(5).iterrows():
            direction = "↑" if r["rank_change"] > 0 else "↓"
            print(f"    {r['episode_id']}  {r['character']:<22} {direction}{abs(r['rank_change'])} ranks")

    print(f"\n  Output files in {out_dir}/:")
    for f in ["suits_episode_interactions.csv", "suits_centrality.csv",
              "suits_influence.csv", "suits_power_trajectory.csv", "suits_dominance.csv"]:
        path = out_dir / f
        if path.exists():
            rows = pd.read_csv(path).shape[0]
            print(f"    {f:<42} {rows:>7,} rows")

    return {
        "graphs": graphs,
        "G_all": G_all,
        "centrality": centrality_df,
        "influence": influence_df,
        "trajectory": trajectory_df,
        "dominance": dominance_df,
    }


def _h(title: str) -> None:
    print(f"\n{'═' * 58}")
    print(f"  {title}")
    print(f"{'═' * 58}")


def main():
    parser = argparse.ArgumentParser(description="Suits Power Graph Pipeline")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    run(Path(args.data_dir))


if __name__ == "__main__":
    main()
