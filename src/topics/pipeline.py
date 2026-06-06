"""
Topic Evolution Pipeline — Step 5 Orchestrator

Reads:  data/processed/suits_dialogue.csv
Writes:
  suits_topic_keywords.csv    — top 15 keywords per topic
  suits_topic_episodes.csv    — per-episode topic weight distribution
  suits_topic_evolution.csv   — topic weights + rolling averages + dominant topic

Usage:
    python src/topics/pipeline.py
    python src/topics/pipeline.py --n-topics 12
    python src/topics/pipeline.py --use-bertopic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from topics.model import LDATopicModel, BERTopicModel, aggregate_episode_text


def run(
    data_dir: Path,
    n_topics: int = 10,
    use_bertopic: bool = False,
    rolling_window: int = 3,
) -> dict[str, pd.DataFrame]:
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # ── Load dialogue ─────────────────────────────────────────────────────────
    _h("Loading dialogue")
    dialogue_path = processed / "suits_dialogue.csv"
    if not dialogue_path.exists():
        sys.exit("[ERROR] suits_dialogue.csv not found. Run src/pipeline.py first.")

    dialogue_df = pd.read_csv(dialogue_path)
    ep_texts = aggregate_episode_text(dialogue_df)
    n_eps = len(ep_texts)
    print(f"  {n_eps} episodes  |  avg {ep_texts['text'].str.split().str.len().mean():.0f} words/episode")

    if n_eps < 3:
        print(f"\n  ⚠  Only {n_eps} episode(s) — topic modelling needs multiple episodes.")
        print("     Saving placeholder. Re-run after full scrape.")
        _save_placeholder(processed, n_topics)
        return {}

    # ── Fit topic model ───────────────────────────────────────────────────────
    _h(f"Fitting {'BERTopic' if use_bertopic else 'LDA'} ({n_topics} topics)")
    if use_bertopic:
        model = BERTopicModel(n_topics=n_topics)
    else:
        model = LDATopicModel(n_topics=n_topics)

    model.fit(ep_texts["text"].tolist())
    print("  Model fitted.")

    # ── Topic keywords ────────────────────────────────────────────────────────
    keywords_df = model.top_words()
    keywords_df.to_csv(processed / "suits_topic_keywords.csv", index=False)
    print(f"\n  Top keywords per topic:")
    for tid in sorted(keywords_df["topic_id"].unique()):
        top5 = keywords_df[keywords_df["topic_id"] == tid]["keyword"].head(5).tolist()
        label = model.topic_label(tid)
        print(f"    T{tid:02d} | {', '.join(top5):<55} → '{label}'")

    # ── Episode topic distributions ────────────────────────────────────────────
    _h("Computing topic distributions per episode")
    topic_matrix = model.transform(ep_texts["text"].tolist())   # (n_eps, n_topics)

    topic_cols = {f"topic_{i:02d}": topic_matrix[:, i].round(4) for i in range(n_topics)}
    ep_topics = ep_texts[["episode_id", "season", "episode_num"]].copy()
    for col, vals in topic_cols.items():
        ep_topics[col] = vals

    ep_topics["dominant_topic"] = np.argmax(topic_matrix, axis=1)
    ep_topics["dominant_topic_label"] = ep_topics["dominant_topic"].apply(
        lambda t: model.topic_label(t)
    )
    ep_topics.to_csv(processed / "suits_topic_episodes.csv", index=False)

    # ── Rolling averages for smooth trend lines ───────────────────────────────
    _h("Computing topic evolution (rolling averages)")
    evo = ep_topics.copy()
    topic_col_names = [f"topic_{i:02d}" for i in range(n_topics)]
    for col in topic_col_names:
        evo[f"{col}_smooth"] = (
            evo[col].rolling(rolling_window, min_periods=1, center=True).mean().round(4)
        )
    evo.to_csv(processed / "suits_topic_evolution.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    _h("Topic Evolution Summary")
    dom_per_season = (
        ep_topics.groupby("season")["dominant_topic_label"]
        .agg(lambda x: x.value_counts().index[0])
    )
    print("  Dominant topic per season:")
    for season, label in dom_per_season.items():
        count = (
            ep_topics[ep_topics["season"] == season]["dominant_topic_label"] == label
        ).sum()
        print(f"    Season {season}: '{label}'  ({count} episodes)")

    print(f"\n  Files saved to {processed}/")
    for f in ["suits_topic_keywords.csv", "suits_topic_episodes.csv", "suits_topic_evolution.csv"]:
        rows = pd.read_csv(processed / f).shape[0]
        print(f"    {f:<42} {rows:>6} rows")

    return {
        "model": model,
        "episode_topics": ep_topics,
        "evolution": evo,
        "keywords": keywords_df,
    }


def _save_placeholder(processed: Path, n_topics: int) -> None:
    """Create empty CSVs so the dashboard doesn't crash on missing files."""
    cols = ["episode_id", "season", "episode_num"] + \
           [f"topic_{i:02d}" for i in range(n_topics)] + \
           ["dominant_topic", "dominant_topic_label"]
    for fname in ["suits_topic_keywords.csv", "suits_topic_episodes.csv",
                  "suits_topic_evolution.csv"]:
        pd.DataFrame(columns=cols).to_csv(processed / fname, index=False)


def _h(title: str) -> None:
    print(f"\n{'═'*58}\n  {title}\n{'═'*58}")


def main():
    parser = argparse.ArgumentParser(description="Suits Topic Evolution")
    parser.add_argument("--data-dir",     default="data")
    parser.add_argument("--n-topics",     type=int, default=10)
    parser.add_argument("--use-bertopic", action="store_true")
    parser.add_argument("--rolling",      type=int, default=3,
                        help="Smoothing window size")
    args = parser.parse_args()
    run(Path(args.data_dir), n_topics=args.n_topics,
        use_bertopic=args.use_bertopic, rolling_window=args.rolling)


if __name__ == "__main__":
    main()
